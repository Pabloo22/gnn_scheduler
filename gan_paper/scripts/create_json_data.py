import json

from gnn_scheduler.job_shop import load_all_from_benchmark


def determine_reference(instance_name):
    if instance_name.startswith("abz"):
        return "J. Adams, E. Balas, D. Zawack. 'The shifting bottleneck procedure for job shop scheduling.', Management Science, Vol. 34, Issue 3, pp. 391-401, 1988."
    elif instance_name in ("ft06", "ft10", "ft20"):
        return "J.F. Muth, G.L. Thompson. 'Industrial scheduling.', Englewood Cliffs, NJ, Prentice-Hall, 1963."
    elif instance_name.startswith("la"):
        return "S. Lawrence. 'Resource constrained project scheduling: an experimental investigation of heuristic scheduling techniques (Supplement).', Graduate School of Industrial Administration. Pittsburgh, Pennsylvania, Carnegie-Mellon University, 1984."
    elif instance_name.startswith("orb"):
        return "D. Applegate, W. Cook. 'A computational study of job-shop scheduling.', ORSA Journal on Computing, Vol. 3, Issue 2, pp. 149-156, 1991."
    elif instance_name.startswith("swv"):
        return "R.H. Storer, S.D. Wu, R. Vaccari. 'New search spaces for sequencing problems with applications to job-shop scheduling.', Management Science Vol. 38, Issue 10, pp. 1495-1509, 1992."
    elif instance_name.startswith("yn"):
        return "T. Yamada, R. Nakano. 'A genetic algorithm applicable to large-scale job-shop problems.', Proceedings of the Second international workshop on parallel problem solving from Nature (PPSN'2). Brussels (Belgium), pp. 281-290, 1992."
    elif instance_name.startswith("ta"):
        return "E. Taillard. 'Benchmarks for basic scheduling problems', European Journal of Operational Research, Vol. 64, Issue 2, pp. 278-285, 1993."
    else:
        return "Unknown Reference"


def main():
    instances = load_all_from_benchmark()
    for instance in instances:
        reference = determine_reference(instance.name)
        instance_dict = instance.to_dict()
        if "metadata" not in instance_dict:
            instance_dict["metadata"] = {}
        instance_dict["metadata"]["reference"] = reference
        # Replace the original instance's dictionary with the updated one
        # This step may need adjustment based on your actual implementation

    instance_dicts = [instance.to_dict() for instance in instances]

    json_file = {}
    for instance_dict in instance_dicts:
        json_file[instance_dict["name"]] = instance_dict

    # Save the instances to a JSON file
    with open("benchmark_instances.json", "w", encoding="utf-8") as f:
        json.dump(json_file, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
