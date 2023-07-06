import pytest
import numpy as np

from alphazero.envs import Connect2, Connect2State


@pytest.mark.parametrize("state, expected_legal_moves", [
    (np.array([1, -1, 0, 0], dtype=np.int8), np.array([2, 3])),
    (np.array([0, 0, 0, 0], dtype=np.int8), np.array([0, 1, 2, 3])),
    (np.array([1, 1, 1, 1], dtype=np.int8), np.array([])),
])
def test_legal_moves(state, expected_legal_moves):
    env = Connect2()
    env.current_state = state
    assert np.array_equal(env.legal_moves(env.current_state),
                          expected_legal_moves)


@pytest.mark.parametrize("state, action, expected_result", [
    (np.array([1, -1, 0, 0], dtype=np.int8), 2, True),
    (np.array([1, -1, 0, 0], dtype=np.int8), 1, False),
    (np.array([1, -1, 0, 0], dtype=np.int8), 4, ValueError),
])
def test_check_action(state, action, expected_result):
    env = Connect2()
    env.current_state = state
    if isinstance(expected_result, type) and issubclass(expected_result,
                                                        Exception):
        with pytest.raises(expected_result):
            env.check_action(env.current_state, action)
    else:
        assert env.check_action(env.current_state, action) == expected_result


@pytest.mark.parametrize("state, expected_result", [
    (np.array([1, 1, -1, -1], dtype=np.int8), True),
    (np.array([1, -1, 1, -1], dtype=np.int8), False),
    (np.array([-1, -1, 1, 1], dtype=np.int8), True),
])
def test_is_winning_state(state, expected_result):
    env = Connect2()
    env.current_state = state
    assert env.is_winning_state(env.current_state) == expected_result


TEST_CASES_STEP = {
    "case1": {
        "initial_state": np.array([0, 0, 0, 0], dtype=np.int8),
        "action": 0,
        "expected_state": np.array([1, 0, 0, 0], dtype=np.int8),
        "expected_reward": 0,
        "expected_terminated": False,
        "expected_current_player": -1
    },
    "case2": {
        "initial_state": np.array([1, 0, 0, 0], dtype=np.int8),
        "action": 1,
        "expected_state": np.array([1, -1, 0, 0], dtype=np.int8),
        "expected_reward": 0,
        "expected_terminated": False,
        "expected_current_player": 1
    },
    "case3": {
        "initial_state": np.array([1, -1, 0, 0], dtype=np.int8),
        "action": 2,
        "expected_state": np.array([1, -1, 1, 0], dtype=np.int8),
        "expected_reward": 0,
        "expected_terminated": False,
        "expected_current_player": -1
    },
    "case4": {
        "initial_state": np.array([1, -1, 1, 0], dtype=np.int8),
        "action": 3,
        "expected_state": np.array([1, -1, 1, -1], dtype=np.int8),
        "expected_reward": 0,
        "expected_terminated": True,
        "expected_current_player": 1
    },
    "case5": {
        "initial_state": np.array([-1, 0, 0, 1], dtype=np.int8),
        "action": 2,
        "expected_state": np.array([-1, 0, 1, 1], dtype=np.int8),
        "expected_reward": 1,
        "expected_terminated": True,
        "expected_current_player": -1
    },
    "case6": {
        "initial_state": np.array([1, -1, 0, 1], dtype=np.int8),
        "action": 2,
        "expected_state": np.array([1, -1, -1, 1], dtype=np.int8),
        "expected_reward": -1,
        "expected_terminated": True,
        "expected_current_player": 1
    },
}

LIST_OF_TEST_CASES_STEP = []


for case, values in TEST_CASES_STEP.items():
    LIST_OF_TEST_CASES_STEP.append(
        pytest.param(
            values["initial_state"],
            values["action"],
            values["expected_state"],
            values["expected_reward"],
            values["expected_terminated"],
            values["expected_current_player"],
            id=case,
        )
    )


@pytest.mark.parametrize(
    "initial_state, action, expected_state, expected_reward, "
    "expected_terminated, expected_current_player",
    LIST_OF_TEST_CASES_STEP)
def test_step(initial_state: Connect2State,
              action: int,
              expected_state: Connect2State,
              expected_reward: float,
              expected_terminated: bool,
              expected_current_player: int):
    env = Connect2()
    env.current_state = initial_state
    num_chips = np.sum(np.abs(initial_state))
    env.current_player = 1 if num_chips % 2 == 0 else -1
    state, reward, terminated, _, info = env.step(action)
    assert np.array_equal(state, expected_state)
    assert reward == expected_reward
    assert terminated == expected_terminated
    assert info["current_player"] == expected_current_player


@pytest.mark.parametrize(
    "num_steps, expected_current_player",
    [(0, 1), (1, -1), (2, 1), (3, -1), (4, 1)]
)
def test_expected_current_player(num_steps, expected_current_player):
    env = Connect2()
    for _ in range(num_steps):
        legal_moves = env.legal_moves(env.current_state)
        env.step(min(legal_moves))
    assert env.current_player == expected_current_player


def test_reset():
    env = Connect2()
    env.step(0)
    env.step(1)

    initial_state, info = env.reset(seed=1)
    assert np.array_equal(initial_state, np.array([0, 0, 0, 0], dtype=np.int8))
    assert env.current_player == 1
    assert info["current_player"] == 1


if __name__ == "__main__":
    pytest.main()
