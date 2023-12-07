from __future__ import annotations

from typing import Optional
import time

from ortools.sat.python import cp_model

from gnn_scheduler.jssp import JobShopInstance


class CPSolver:
    def __init__(self,
                 job_shop_instance: JobShopInstance,
                 log_search_progress: bool = False,
                 time_limit: Optional[float] = None):
        self.instance = job_shop_instance
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.solver.parameters.log_search_progress = log_search_progress
        if time_limit is not None:
            self.solver.parameters.max_time_in_seconds = time_limit
        self.operations_start = {}
        self.makespan = None

    def _create_variables(self):
        """Creates two variables for each operation: start and end time."""
        for j, job in enumerate(self.instance.jobs):
            for p, operation in enumerate(job):
                op_id = operation.get_id(j, p)
                start_var = self.model.NewIntVar(
                    0, self.instance.total_duration, f"start_{op_id}"
                )
                end_var = self.model.NewIntVar(
                    0, self.instance.total_duration, f"end_{op_id}"
                )
                self.operations_start[op_id] = (start_var, end_var)
                self.model.Add(end_var == start_var + operation.duration)

    def _add_constraints(self):
        """Adds job and machine constraints.

        Job Constraints: Ensure that operations within a job are performed in
        sequence. If operation A must precede operation B in a job, we ensure
        A's end time is less than or equal to B's start time.

        Machine Constraints: Operations assigned to the same machine cannot
        overlap. This is ensured by creating interval variables (which
        represent the duration an operation occupies a machine)
        and adding a 'no overlap' constraint for these intervals on
        each machine.
        """
        # Job constraints: operations in a job must be done in order
        for job_id, job in enumerate(self.instance.jobs):
            for position in range(1, len(job)):
                self.model.Add(
                    self.operations_start[job[position - 1].get_id(job_id, position - 1)][1]
                    <= self.operations_start[job[position].get_id(job_id, position)][0]
                )

        # Machine constraints: operations on the same machine cannot overlap
        machines_operations = [[] for _ in range(self.instance.n_machines)]
        for j, job in enumerate(self.instance.jobs):
            for p, operation in enumerate(job):
                machines_operations[operation.machine_id].append(
                    (self.operations_start[operation.get_id(j, p)], 
                     operation.duration)
                )
        for machine_id, operations in enumerate(machines_operations):
            intervals = []
            for (start_var, end_var), duration in operations:
                interval_var = self.model.NewIntervalVar(
                    start_var, duration, end_var, f"interval_{machine_id}"
                )
                intervals.append(interval_var)
            self.model.AddNoOverlap(intervals)

    def _set_objective(self):
        """The objective is to minimize the makespan, which is the total
        duration of the schedule."""
        self.makespan = self.model.NewIntVar(
            0, self.instance.total_duration, "makespan"
        )
        end_times = [end for _, end in self.operations_start.values()]
        self.model.AddMaxEquality(self.makespan, end_times)
        self.model.Minimize(self.makespan)

    def solve(self) -> dict[str, int] | None:
        """Creates the variables, constraints and objective, and solves the
        problem.

        If a solution is found, it extracts and returns the start times of
        each operation and the makespan. If no solution is found, it returns
        None.
        """
        self._create_variables()
        self._add_constraints()
        self._set_objective()

        # Solve the problem
        start_time = time.perf_counter()
        status = self.solver.Solve(self.model)
        elapsed_time = time.perf_counter() - start_time
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # Retrieve solution
            solution = {
                op_id: self.solver.Value(start_var)
                for op_id, (start_var, _) in self.operations_start.items()
            }
            solution["makespan"] = self.solver.Value(self.makespan)
            solution["elapsed_time"] = elapsed_time
            status = "optimal" if status == cp_model.OPTIMAL else "feasible"
            solution["status"] = status
            return solution


if __name__ == "__main__":
    from gnn_scheduler.jssp import load_from_benchmark

    instance = load_from_benchmark("swv10")
    solver = CPSolver(instance, time_limit=1)
    solution_ = solver.solve()
    print(solution_)
