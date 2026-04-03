from __future__ import annotations

from itertools import product, permutations
from typing import Iterable

import numpy as np


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


class DynamicPricingEnv:
    def __init__(self, *, horizon: int = 6, elasticity_shift: float = 0.0) -> None:
        self.horizon = horizon
        self.elasticity_shift = elasticity_shift
        self.n_states = self.horizon * 3 * 4
        self.n_actions = 3
        self._state = (1, 3, 0)

    def encode(self, demand: int, inventory: int, time_index: int) -> int:
        return ((time_index * 3) + demand) * 4 + inventory

    def decode(self, state: int) -> tuple[int, int, int]:
        inventory = state % 4
        state //= 4
        demand = state % 3
        time_index = state // 3
        return demand, inventory, time_index

    def reset(self, rng: np.random.Generator) -> int:
        self._state = (int(rng.integers(0, 3)), int(rng.integers(2, 4)), 0)
        return self.encode(*self._state)

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool, dict[str, float]]:
        demand, inventory, time_index = self._state
        price = [7.0, 10.0, 13.0][action]
        base_prob = 0.78 + 0.11 * (demand - 1) - 0.17 * action + self.elasticity_shift
        sell_prob = float(np.clip(base_prob + 0.04 * (time_index % 2), 0.05, 0.95))
        sold = 0
        if inventory > 0 and rng.random() < sell_prob:
            sold = 1
        if inventory > 1 and action == 0 and demand == 2 and rng.random() < 0.35:
            sold += 1
        sold = min(sold, inventory)
        next_inventory = inventory - sold
        reward = price * sold
        drift = int(rng.choice([-1, 0, 1], p=[0.2 + 0.03 * action, 0.55, 0.25 - 0.03 * action]))
        next_demand = _clamp(demand + drift, 0, 2)
        done = next_inventory == 0 or time_index + 1 >= self.horizon
        if not done:
            self._state = (next_demand, next_inventory, time_index + 1)
            next_state = self.encode(*self._state)
        else:
            next_state = self.encode(next_demand, next_inventory, min(time_index, self.horizon - 1))
        return next_state, reward, done, {"success": float(next_inventory == 0), "sales": float(sold)}


class SupplyChainRoutingEnv:
    def __init__(self, *, horizon: int = 6) -> None:
        self.horizon = horizon
        self.n_states = self.horizon * 3 * 3 * 3
        self.n_actions = 3
        self._state = (0, 2, 1, 0)

    def encode(self, stage: int, buffer: int, disruption: int, time_index: int) -> int:
        return (((time_index * 3) + stage) * 3 + buffer) * 3 + disruption

    def decode(self, state: int) -> tuple[int, int, int, int]:
        disruption = state % 3
        state //= 3
        buffer = state % 3
        state //= 3
        stage = state % 3
        time_index = state // 3
        return stage, buffer, disruption, time_index

    def reset(self, rng: np.random.Generator) -> int:
        self._state = (0, int(rng.integers(1, 3)), int(rng.integers(0, 3)), 0)
        return self.encode(*self._state)

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool, dict[str, float]]:
        stage, buffer, disruption, time_index = self._state
        costs = [1.0, 1.5, 2.3]
        delay_prob = [0.38, 0.2, 0.08][action] + 0.1 * disruption
        delayed = float(rng.random() < min(delay_prob, 0.9))
        progress = 0 if delayed else 1
        if action == 2 and not delayed and stage == 0 and rng.random() < 0.35:
            progress = 2
        next_stage = min(2, stage + progress)
        supply_change = 1 if action == 1 and rng.random() < 0.45 else 0
        next_buffer = _clamp(buffer + supply_change - 1, 0, 2)
        stockout = float(next_buffer == 0)
        next_disruption = _clamp(disruption + int(rng.choice([-1, 0, 1], p=[0.2, 0.55, 0.25])), 0, 2)
        reward = -costs[action] - 1.4 * delayed - 1.8 * stockout + 8.0 * float(next_stage == 2)
        done = next_stage == 2 or time_index + 1 >= self.horizon
        if not done:
            self._state = (next_stage, next_buffer, next_disruption, time_index + 1)
            next_state = self.encode(*self._state)
        else:
            next_state = self.encode(next_stage, next_buffer, next_disruption, min(time_index, self.horizon - 1))
        return next_state, reward, done, {
            "success": float(next_stage == 2),
            "violation": stockout,
            "delay": delayed,
            "progress": float(progress),
        }


class TrafficGridEnv:
    def __init__(self, *, horizon: int = 8) -> None:
        self.horizon = horizon
        self.n_states = self.horizon * (3 ** 4)
        self.n_actions = 4
        self._state = (1, 1, 1, 1, 0)

    def encode(self, q1_ns: int, q1_ew: int, q2_ns: int, q2_ew: int, time_index: int) -> int:
        queues = (((q1_ns * 3) + q1_ew) * 3 + q2_ns) * 3 + q2_ew
        return time_index * (3 ** 4) + queues

    def decode(self, state: int) -> tuple[int, int, int, int, int]:
        time_index = state // (3 ** 4)
        queues = state % (3 ** 4)
        q2_ew = queues % 3
        queues //= 3
        q2_ns = queues % 3
        queues //= 3
        q1_ew = queues % 3
        q1_ns = queues // 3
        return q1_ns, q1_ew, q2_ns, q2_ew, time_index

    def reset(self, rng: np.random.Generator) -> int:
        self._state = tuple(int(rng.integers(0, 2)) for _ in range(4)) + (0,)
        return self.encode(*self._state)

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool, dict[str, float]]:
        q1_ns, q1_ew, q2_ns, q2_ew, time_index = self._state
        phase_1 = action // 2
        phase_2 = action % 2
        served_1 = 1 if phase_1 == 0 and q1_ns > 0 else 1 if phase_1 == 1 and q1_ew > 0 else 0
        served_2 = 1 if phase_2 == 0 and q2_ns > 0 else 1 if phase_2 == 1 and q2_ew > 0 else 0
        q1_ns = max(0, q1_ns - int(phase_1 == 0 and q1_ns > 0))
        q1_ew = max(0, q1_ew - int(phase_1 == 1 and q1_ew > 0))
        q2_ns = max(0, q2_ns - int(phase_2 == 0 and q2_ns > 0))
        q2_ew = max(0, q2_ew - int(phase_2 == 1 and q2_ew > 0))
        arrivals = [
            int(rng.random() < (0.65 if time_index % 2 == 0 else 0.35)),
            int(rng.random() < (0.35 if time_index % 2 == 0 else 0.65)),
            int(rng.random() < (0.35 if time_index % 2 == 0 else 0.6)),
            int(rng.random() < (0.6 if time_index % 2 == 0 else 0.35)),
        ]
        next_state_tuple = (
            _clamp(q1_ns + arrivals[0], 0, 2),
            _clamp(q1_ew + arrivals[1], 0, 2),
            _clamp(q2_ns + arrivals[2], 0, 2),
            _clamp(q2_ew + arrivals[3], 0, 2),
            time_index + 1,
        )
        total_queue = sum(next_state_tuple[:4])
        throughput = served_1 + served_2
        reward = float(throughput) - 0.5 * total_queue
        done = time_index + 1 >= self.horizon
        if not done:
            self._state = next_state_tuple
            next_state = self.encode(*self._state)
        else:
            next_state = self.encode(*next_state_tuple[:-1], self.horizon - 1)
        return next_state, reward, done, {
            "success": float(done and total_queue <= 1),
            "throughput": float(throughput),
            "queue": float(total_queue),
        }


class EnergyGridEnv:
    def __init__(self, *, horizon: int = 6) -> None:
        self.horizon = horizon
        self.n_states = self.horizon * 3 * 3 * 3
        self.n_actions = 4
        self._state = (1, 1, 1, 0)

    def encode(self, demand: int, battery: int, renewable: int, time_index: int) -> int:
        return (((time_index * 3) + demand) * 3 + battery) * 3 + renewable

    def decode(self, state: int) -> tuple[int, int, int, int]:
        renewable = state % 3
        state //= 3
        battery = state % 3
        state //= 3
        demand = state % 3
        time_index = state // 3
        return demand, battery, renewable, time_index

    def reset(self, rng: np.random.Generator) -> int:
        self._state = (int(rng.integers(0, 3)), int(rng.integers(1, 3)), int(rng.integers(0, 3)), 0)
        return self.encode(*self._state)

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool, dict[str, float]]:
        demand, battery, renewable, time_index = self._state
        demand_units = demand + 1
        dispatch_cap = [0, 1, 2, 0][action]
        battery_use = min(battery, dispatch_cap, max(demand_units - renewable, 0))
        grid_need = max(demand_units - renewable - battery_use, 0)
        if action == 0 and grid_need > 0:
            grid_use = max(grid_need - 1, 0)
        else:
            grid_use = grid_need
        served = min(demand_units, renewable + battery_use + grid_use)
        unmet = demand_units - served
        battery_recharge = max(renewable - min(renewable, demand_units), 0)
        next_battery = _clamp(battery - battery_use + battery_recharge, 0, 2)
        next_demand = _clamp(demand + int(rng.choice([-1, 0, 1], p=[0.25, 0.45, 0.3])), 0, 2)
        next_renewable = _clamp(renewable + int(rng.choice([-1, 0, 1], p=[0.35, 0.35, 0.3])), 0, 2)
        reward = 0.7 * served - 1.2 * grid_use - 2.8 * unmet - 0.15 * abs(next_battery - 1)
        done = time_index + 1 >= self.horizon
        if not done:
            self._state = (next_demand, next_battery, next_renewable, time_index + 1)
            next_state = self.encode(*self._state)
        else:
            next_state = self.encode(next_demand, next_battery, next_renewable, self.horizon - 1)
        return next_state, reward, done, {
            "violation": float(unmet > 0),
            "blackout": float(unmet),
            "served": float(served),
        }


class TradingExecutionEnv:
    def __init__(self, *, horizon: int = 5) -> None:
        self.horizon = horizon
        self.n_states = self.horizon * 5 * 3
        self.n_actions = 3
        self._state = (4, 1, 0)

    def encode(self, remaining: int, volatility: int, time_index: int) -> int:
        return ((time_index * 5) + remaining) * 3 + volatility

    def decode(self, state: int) -> tuple[int, int, int]:
        volatility = state % 3
        state //= 3
        remaining = state % 5
        time_index = state // 5
        return remaining, volatility, time_index

    def reset(self, rng: np.random.Generator) -> int:
        self._state = (4, int(rng.integers(0, 3)), 0)
        return self.encode(*self._state)

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool, dict[str, float]]:
        remaining, volatility, time_index = self._state
        desired = [0, 1, 2][action]
        executed = min(remaining, desired)
        if action == 0 and remaining > 0 and rng.random() < 0.3:
            executed = 1
        slippage = [0.0, 0.6, 1.2][action] * (1.0 + 0.35 * volatility)
        reward = 2.0 * executed - slippage * executed - 0.25 * remaining * volatility
        next_remaining = remaining - executed
        next_volatility = _clamp(volatility + int(rng.choice([-1, 0, 1], p=[0.25, 0.45, 0.3])), 0, 2)
        done = next_remaining == 0 or time_index + 1 >= self.horizon
        if done and next_remaining > 0:
            reward -= 2.5 * next_remaining
        if done and next_remaining == 0:
            reward += 2.0
        if not done:
            self._state = (next_remaining, next_volatility, time_index + 1)
            next_state = self.encode(*self._state)
        else:
            next_state = self.encode(next_remaining, next_volatility, min(time_index, self.horizon - 1))
        return next_state, reward, done, {
            "success": float(next_remaining == 0),
            "executed": float(executed),
            "inventory_left": float(next_remaining),
        }


class ChemicalSafetyEnv:
    def __init__(self, *, horizon: int = 6) -> None:
        self.horizon = horizon
        self.n_states = self.horizon * 5 * 5 * 3
        self.n_actions = 4
        self._state = (2, 2, 1, 0)

    def encode(self, temperature: int, pressure: int, concentration: int, time_index: int) -> int:
        return (((time_index * 5) + temperature) * 5 + pressure) * 3 + concentration

    def decode(self, state: int) -> tuple[int, int, int, int]:
        concentration = state % 3
        state //= 3
        pressure = state % 5
        state //= 5
        temperature = state % 5
        time_index = state // 5
        return temperature, pressure, concentration, time_index

    def reset(self, rng: np.random.Generator) -> int:
        self._state = (int(rng.integers(1, 4)), int(rng.integers(1, 4)), int(rng.integers(0, 3)), 0)
        return self.encode(*self._state)

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool, dict[str, float]]:
        temperature, pressure, concentration, time_index = self._state
        temp_shift = [-1, 0, 1, 0][action]
        pressure_shift = [-1, 0, 1, -2][action]
        conc_shift = [0, 0, 1, -1][action]
        next_temperature = _clamp(temperature + temp_shift + int(rng.choice([-1, 0, 1], p=[0.15, 0.7, 0.15])), 0, 4)
        next_pressure = _clamp(pressure + pressure_shift + int(rng.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])), 0, 4)
        next_concentration = _clamp(concentration + conc_shift, 0, 2)
        violation = float(next_temperature in (0, 4) or next_pressure in (0, 4))
        target_quality = 1.0 if next_concentration == 1 else 0.4 if next_concentration == 2 else -0.2
        reward = target_quality - 0.35 * abs(next_temperature - 2) - 0.3 * abs(next_pressure - 2) - 3.2 * violation
        catastrophic = bool(violation and rng.random() < 0.4)
        done = catastrophic or time_index + 1 >= self.horizon
        if not done:
            self._state = (next_temperature, next_pressure, next_concentration, time_index + 1)
            next_state = self.encode(*self._state)
        else:
            next_state = self.encode(next_temperature, next_pressure, next_concentration, min(time_index, self.horizon - 1))
        return next_state, reward, done, {
            "violation": violation,
            "success": float(done and not violation and next_concentration == 1),
            "quality": target_quality,
        }


class JobShopCurriculumEnv:
    def __init__(self, *, difficulty: int = 0, max_horizon: int = 10) -> None:
        self.difficulty = difficulty
        self.max_horizon = max_horizon
        self.horizon = [6, 8, 10][difficulty]
        self.n_states = self.max_horizon * 4 * 4 * 3 * 2
        self.n_actions = 3
        self._state = (1, 1, 0, 0, 0)

    def encode(self, short_left: int, long_left: int, busy: int, current_long: int, time_index: int) -> int:
        return ((((time_index * 4) + short_left) * 4 + long_left) * 3 + busy) * 2 + current_long

    def decode(self, state: int) -> tuple[int, int, int, int, int]:
        current_long = state % 2
        state //= 2
        busy = state % 3
        state //= 3
        long_left = state % 4
        state //= 4
        short_left = state % 4
        time_index = state // 4
        return short_left, long_left, busy, current_long, time_index

    def reset(self, rng: np.random.Generator) -> int:
        short_left = [1, 2, 2][self.difficulty]
        long_left = [1, 1, 2][self.difficulty]
        self._state = (short_left, long_left, 0, 0, 0)
        return self.encode(*self._state)

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool, dict[str, float]]:
        short_left, long_left, busy, current_long, time_index = self._state
        reward = -0.2
        next_short = short_left
        next_long = long_left
        next_busy = busy
        next_current_long = current_long
        if busy > 0:
            next_busy = busy - 1
            if next_busy == 0:
                reward += 3.0 if current_long else 1.8
            if action in (0, 1):
                reward -= 0.25
        elif action == 0:
            if short_left > 0:
                next_short -= 1
                next_busy = 1
                next_current_long = 0
                reward += 0.1
            else:
                reward -= 1.0
        elif action == 1:
            if long_left > 0:
                next_long -= 1
                next_busy = 2
                next_current_long = 1
            else:
                reward -= 1.0
        else:
            reward -= 0.35 if (short_left + long_left) > 0 else 0.05
        done = (next_short + next_long == 0 and next_busy == 0) or time_index + 1 >= self.horizon
        if done and (next_short + next_long + next_busy) > 0:
            reward -= 2.0 * (next_short + next_long + next_busy)
        if not done:
            self._state = (next_short, next_long, next_busy, next_current_long, time_index + 1)
            next_state = self.encode(*self._state)
        else:
            next_state = self.encode(next_short, next_long, next_busy, next_current_long, min(time_index, self.max_horizon - 1))
        return next_state, reward, done, {
            "success": float(done and (next_short + next_long + next_busy) == 0),
            "remaining_jobs": float(next_short + next_long + next_busy),
        }


class HierarchicalLogisticsEnv:
    def __init__(self, *, horizon: int = 7) -> None:
        self.horizon = horizon
        self.n_states = self.horizon * 4 * 3 * 2
        self.n_actions = 4
        self._state = (0, 1, 0, 0)

    def encode(self, region: int, backlog: int, disruption: int, time_index: int) -> int:
        return (((time_index * 4) + region) * 3 + backlog) * 2 + disruption

    def decode(self, state: int) -> tuple[int, int, int, int]:
        disruption = state % 2
        state //= 2
        backlog = state % 3
        state //= 3
        region = state % 4
        time_index = state // 4
        return region, backlog, disruption, time_index

    def reset(self, rng: np.random.Generator) -> int:
        self._state = (0, int(rng.integers(0, 2)), int(rng.integers(0, 2)), 0)
        return self.encode(*self._state)

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool, dict[str, float]]:
        region, backlog, disruption, time_index = self._state
        delay_chance = [0.28, 0.16, 0.08, 0.4][action] + 0.08 * disruption
        delayed = float(rng.random() < min(delay_chance, 0.9))
        progress = 0
        if action == 2 and not delayed:
            progress = 2
        elif action in (0, 1) and not delayed:
            progress = 1
        next_region = min(3, region + progress)
        backlog_delta = 1 if action == 3 else -1 if action == 1 and backlog > 0 else 0
        next_backlog = _clamp(backlog + backlog_delta + int(rng.choice([-1, 0, 1], p=[0.15, 0.6, 0.25])), 0, 2)
        next_disruption = _clamp(disruption + int(rng.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])), 0, 1)
        costs = [0.7, 1.0, 2.1, 0.2]
        violation = float(next_backlog == 2)
        reward = 1.5 * progress - costs[action] - 0.9 * next_backlog - 1.1 * delayed + 6.0 * float(next_region == 3)
        done = next_region == 3 or time_index + 1 >= self.horizon
        if not done:
            self._state = (next_region, next_backlog, next_disruption, time_index + 1)
            next_state = self.encode(*self._state)
        else:
            next_state = self.encode(next_region, next_backlog, next_disruption, self.horizon - 1)
        return next_state, reward, done, {
            "success": float(next_region == 3),
            "violation": violation,
            "progress": float(progress),
        }


class HvacEnv:
    def __init__(self, *, horizon: int = 6) -> None:
        self.horizon = horizon
        self.n_states = self.horizon * 5 * 3 * 2 * 3
        self.n_actions = 4
        self._state = (2, 1, 1, 1, 0)

    def encode(self, indoor: int, outdoor: int, occupied: int, tariff: int, time_index: int) -> int:
        return ((((time_index * 5) + indoor) * 3 + outdoor) * 2 + occupied) * 3 + tariff

    def decode(self, state: int) -> tuple[int, int, int, int, int]:
        tariff = state % 3
        state //= 3
        occupied = state % 2
        state //= 2
        outdoor = state % 3
        state //= 3
        indoor = state % 5
        time_index = state // 5
        return indoor, outdoor, occupied, tariff, time_index

    def reset(self, rng: np.random.Generator) -> int:
        self._state = (int(rng.integers(1, 4)), int(rng.integers(0, 3)), int(rng.integers(0, 2)), int(rng.integers(0, 3)), 0)
        return self.encode(*self._state)

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool, dict[str, float]]:
        indoor, outdoor, occupied, tariff, time_index = self._state
        drift = {-1: -1, 0: 0, 1: 1}[outdoor - 1]
        action_shift = {0: 0, 1: -1, 2: -2, 3: 1}[action]
        next_indoor = _clamp(indoor + drift + action_shift + int(rng.choice([-1, 0, 1], p=[0.1, 0.75, 0.15])), 0, 4)
        next_outdoor = _clamp(outdoor + int(rng.choice([-1, 0, 1], p=[0.25, 0.45, 0.3])), 0, 2)
        next_occupied = int(rng.random() < (0.65 if 1 <= time_index <= 3 else 0.35))
        next_tariff = _clamp(tariff + int(rng.choice([-1, 0, 1], p=[0.2, 0.45, 0.35])), 0, 2)
        target = 2 if occupied else 1
        energy_use = [0.0, 1.0, 2.0, 1.4][action]
        comfort_penalty = abs(next_indoor - target) * (2.0 if occupied else 0.8)
        reward = -comfort_penalty - energy_use * (1.0 + 0.45 * tariff)
        done = time_index + 1 >= self.horizon
        if not done:
            self._state = (next_indoor, next_outdoor, next_occupied, next_tariff, time_index + 1)
            next_state = self.encode(*self._state)
        else:
            next_state = self.encode(next_indoor, next_outdoor, next_occupied, next_tariff, self.horizon - 1)
        return next_state, reward, done, {
            "success": float(done and comfort_penalty <= 1.0),
            "violation": float(comfort_penalty > 2.0),
            "comfort_penalty": float(comfort_penalty),
        }


class EnvelopeControlEnv:
    def __init__(self, *, horizon: int = 6) -> None:
        self.horizon = horizon
        self.n_states = self.horizon * 5 * 5 * 3
        self.n_actions = 4
        self._state = (4, 1, 1, 0)

    def encode(self, distance: int, speed: int, load: int, time_index: int) -> int:
        return (((time_index * 5) + distance) * 5 + speed) * 3 + load

    def decode(self, state: int) -> tuple[int, int, int, int]:
        load = state % 3
        state //= 3
        speed = state % 5
        state //= 5
        distance = state % 5
        time_index = state // 5
        return distance, speed, load, time_index

    def safe_speed(self, distance: int, load: int) -> int:
        return _clamp(distance - load + 1, 1, 4)

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool, dict[str, float]]:
        distance, speed, load, time_index = self._state
        accel = [-1, 0, 1, -2][action]
        next_speed = _clamp(speed + accel + int(rng.choice([-1, 0, 1], p=[0.15, 0.7, 0.15])), 0, 4)
        progress = max(next_speed - 1, 0)
        next_distance = _clamp(distance - progress, 0, 4)
        next_load = _clamp(load + int(rng.choice([-1, 0, 1], p=[0.2, 0.55, 0.25])), 0, 2)
        violation = float(next_speed > self.safe_speed(distance, load))
        reward = float(progress) - 0.25 * abs(action - 1) - 4.0 * violation - 0.2 * abs(next_speed - self.safe_speed(distance, load))
        done = next_distance == 0 or time_index + 1 >= self.horizon
        if not done:
            self._state = (next_distance, next_speed, next_load, time_index + 1)
            next_state = self.encode(*self._state)
        else:
            next_state = self.encode(next_distance, next_speed, next_load, self.horizon - 1)
        return next_state, reward, done, {
            "success": float(next_distance == 0 and not violation),
            "violation": violation,
            "safe_speed": float(self.safe_speed(distance, load)),
        }

    def reset(self, rng: np.random.Generator) -> int:
        self._state = (4, int(rng.integers(0, 3)), int(rng.integers(0, 3)), 0)
        return self.encode(*self._state)


class SimTransferEnv:
    def __init__(self, *, horizon: int = 6, reality_gap: float = 0.0) -> None:
        self.horizon = horizon
        self.reality_gap = reality_gap
        self.n_states = self.horizon * 5 * 5 * 3
        self.n_actions = 3
        self._state = (2, 2, 1, 0)

    def encode(self, offset: int, velocity: int, terrain: int, time_index: int) -> int:
        return (((time_index * 5) + offset) * 5 + velocity) * 3 + terrain

    def decode(self, state: int) -> tuple[int, int, int, int]:
        terrain = state % 3
        state //= 3
        velocity = state % 5
        state //= 5
        offset = state % 5
        time_index = state // 5
        return offset, velocity, terrain, time_index

    def reset(self, rng: np.random.Generator) -> int:
        self._state = (int(rng.integers(0, 5)), int(rng.integers(1, 4)), int(rng.integers(0, 3)), 0)
        return self.encode(*self._state)

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool, dict[str, float]]:
        offset, velocity, terrain, time_index = self._state
        signed_offset = offset - 2
        signed_velocity = velocity - 2
        control = action - 1
        gain = 1.0 + self.reality_gap + 0.1 * terrain
        velocity_next = signed_velocity * 0.45 + control * gain - 0.25 * np.sign(signed_offset)
        offset_next = signed_offset + velocity_next
        next_signed_velocity = _clamp(int(round(velocity_next)), -2, 2)
        next_signed_offset = _clamp(int(round(offset_next)), -2, 2)
        next_terrain = _clamp(terrain + int(rng.choice([-1, 0, 1], p=[0.2, 0.55, 0.25])), 0, 2)
        reward = -abs(next_signed_offset) - 0.35 * abs(next_signed_velocity) - 0.1 * abs(control)
        done = (next_signed_offset == 0 and next_signed_velocity == 0) or time_index + 1 >= self.horizon
        if done and next_signed_offset == 0 and next_signed_velocity == 0:
            reward += 2.5
        if not done:
            self._state = (next_signed_offset + 2, next_signed_velocity + 2, next_terrain, time_index + 1)
            next_state = self.encode(*self._state)
        else:
            next_state = self.encode(next_signed_offset + 2, next_signed_velocity + 2, next_terrain, self.horizon - 1)
        return next_state, reward, done, {
            "success": float(next_signed_offset == 0 and next_signed_velocity == 0),
            "gap": float(self.reality_gap),
        }


class ResourceAllocationEnv:
    def __init__(self, *, horizon: int = 6) -> None:
        self.horizon = horizon
        self.n_states = self.horizon * 4 * 3 * 3 * 2
        self.n_actions = 3
        self._state = (1, 1, 1, 0, 0)

    def encode(self, queue: int, cpu: int, memory: int, priority: int, time_index: int) -> int:
        return ((((time_index * 4) + queue) * 3 + cpu) * 3 + memory) * 2 + priority

    def decode(self, state: int) -> tuple[int, int, int, int, int]:
        priority = state % 2
        state //= 2
        memory = state % 3
        state //= 3
        cpu = state % 3
        state //= 3
        queue = state % 4
        time_index = state // 4
        return queue, cpu, memory, priority, time_index

    def reset(self, rng: np.random.Generator) -> int:
        self._state = (int(rng.integers(0, 3)), int(rng.integers(0, 3)), int(rng.integers(0, 3)), int(rng.integers(0, 2)), 0)
        return self.encode(*self._state)

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool, dict[str, float]]:
        queue, cpu, memory, priority, time_index = self._state
        allocation = [1, 2, 3][action]
        demand = queue + priority + 1
        overload = float(action == 2 and (cpu + memory) >= 4)
        served = min(demand, allocation)
        next_queue = _clamp(queue + int(rng.integers(0, 3)) - served // 2, 0, 3)
        next_cpu = _clamp(cpu + int(action == 2) - int(action == 0 and cpu > 0 and rng.random() < 0.35), 0, 2)
        next_memory = _clamp(memory + int(action == 2 and rng.random() < 0.5) - int(action == 0 and memory > 0 and rng.random() < 0.3), 0, 2)
        next_priority = int(rng.random() < 0.4)
        reward = 1.8 * min(served, 2) - abs(demand - allocation) - 1.8 * overload - 0.8 * next_queue
        done = time_index + 1 >= self.horizon
        if not done:
            self._state = (next_queue, next_cpu, next_memory, next_priority, time_index + 1)
            next_state = self.encode(*self._state)
        else:
            next_state = self.encode(next_queue, next_cpu, next_memory, next_priority, self.horizon - 1)
        return next_state, reward, done, {
            "success": float(done and next_queue <= 1 and not overload),
            "violation": overload,
            "served": float(served),
        }


class BlocksWorldEnv:
    blocks = ("A", "B", "C")
    action_pairs = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))

    def __init__(self, *, horizon: int = 6) -> None:
        self.horizon = horizon
        self.states = self._enumerate_states()
        self.state_to_index = {state: index for index, state in enumerate(self.states)}
        self.goal_state = ((), (), ("C", "B", "A"))
        self.goal_index = self.state_to_index[self.goal_state]
        self.goal_distance = self._compute_goal_distances()
        self.n_states = self.horizon * len(self.states)
        self.n_actions = len(self.action_pairs)
        self._state = (self.states[0], 0)

    def _enumerate_states(self) -> list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
        states = set()
        for ordering in permutations(self.blocks):
            for cut_1 in range(4):
                for cut_2 in range(cut_1, 4):
                    states.add((tuple(ordering[:cut_1]), tuple(ordering[cut_1:cut_2]), tuple(ordering[cut_2:])))
        return sorted(states)

    def _next_blocks_state(self, state: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], action: int) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        source, target = self.action_pairs[action]
        if not state[source]:
            return state
        stacks = [list(stack) for stack in state]
        block = stacks[source].pop()
        stacks[target].append(block)
        return tuple(tuple(stack) for stack in stacks)

    def _compute_goal_distances(self) -> dict[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], int]:
        distances = {self.goal_state: 0}
        frontier = [self.goal_state]
        while frontier:
            current = frontier.pop(0)
            current_distance = distances[current]
            for candidate in self.states:
                for action in range(len(self.action_pairs)):
                    if self._next_blocks_state(candidate, action) == current and candidate not in distances:
                        distances[candidate] = current_distance + 1
                        frontier.append(candidate)
        return distances

    def encode(self, state_index: int, time_index: int) -> int:
        return time_index * len(self.states) + state_index

    def decode(self, state: int) -> tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], int]:
        time_index = state // len(self.states)
        state_index = state % len(self.states)
        return self.states[state_index], time_index

    def reset(self, rng: np.random.Generator) -> int:
        non_goal = [state for state in self.states if state != self.goal_state]
        start_state = non_goal[int(rng.integers(0, len(non_goal)))]
        self._state = (start_state, 0)
        return self.encode(self.state_to_index[start_state], 0)

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool, dict[str, float]]:
        blocks_state, time_index = self._state
        next_blocks = self._next_blocks_state(blocks_state, action)
        invalid = float(next_blocks == blocks_state and not blocks_state[self.action_pairs[action][0]])
        reward = -0.2 - invalid
        if next_blocks == self.goal_state:
            reward += 5.0
        done = next_blocks == self.goal_state or time_index + 1 >= self.horizon
        if not done:
            self._state = (next_blocks, time_index + 1)
            next_state = self.encode(self.state_to_index[next_blocks], time_index + 1)
        else:
            next_state = self.encode(self.state_to_index[next_blocks], self.horizon - 1)
        return next_state, reward, done, {
            "success": float(next_blocks == self.goal_state),
            "distance": float(self.goal_distance[next_blocks]),
        }


class MapfEnv:
    moves = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self, *, horizon: int = 8) -> None:
        self.horizon = horizon
        self.n_states = self.horizon * 9 * 9
        self.n_actions = 25
        self.starts = (0, 2)
        self.goals = (6, 8)
        self._state = (self.starts[0], self.starts[1], 0)

    def encode(self, pos_a: int, pos_b: int, time_index: int) -> int:
        return (time_index * 9 + pos_a) * 9 + pos_b

    def decode(self, state: int) -> tuple[int, int, int]:
        pos_b = state % 9
        state //= 9
        pos_a = state % 9
        time_index = state // 9
        return pos_a, pos_b, time_index

    def _to_xy(self, position: int) -> tuple[int, int]:
        return divmod(position, 3)

    def _to_pos(self, x_coord: int, y_coord: int) -> int:
        return x_coord * 3 + y_coord

    def _apply_move(self, position: int, move_index: int) -> int:
        x_coord, y_coord = self._to_xy(position)
        dx, dy = self.moves[move_index]
        next_x = _clamp(x_coord + dx, 0, 2)
        next_y = _clamp(y_coord + dy, 0, 2)
        return self._to_pos(next_x, next_y)

    def manhattan(self, position: int, goal: int) -> int:
        x_1, y_1 = self._to_xy(position)
        x_2, y_2 = self._to_xy(goal)
        return abs(x_1 - x_2) + abs(y_1 - y_2)

    def reset(self, rng: np.random.Generator) -> int:
        self._state = (self.starts[0], self.starts[1], 0)
        return self.encode(*self._state)

    def step(self, action: int, rng: np.random.Generator) -> tuple[int, float, bool, dict[str, float]]:
        pos_a, pos_b, time_index = self._state
        action_a = action // 5
        action_b = action % 5
        next_a = self._apply_move(pos_a, action_a)
        next_b = self._apply_move(pos_b, action_b)
        collision = next_a == next_b or (next_a == pos_b and next_b == pos_a)
        if collision:
            next_a, next_b = pos_a, pos_b
        distance_sum = self.manhattan(next_a, self.goals[0]) + self.manhattan(next_b, self.goals[1])
        reward = -0.2 - 0.12 * distance_sum - 2.0 * float(collision)
        success = next_a == self.goals[0] and next_b == self.goals[1]
        if success:
            reward += 5.0
        done = success or time_index + 1 >= self.horizon
        if not done:
            self._state = (next_a, next_b, time_index + 1)
            next_state = self.encode(*self._state)
        else:
            next_state = self.encode(next_a, next_b, self.horizon - 1)
        return next_state, reward, done, {
            "success": float(success),
            "violation": float(collision),
            "distance": float(distance_sum),
        }