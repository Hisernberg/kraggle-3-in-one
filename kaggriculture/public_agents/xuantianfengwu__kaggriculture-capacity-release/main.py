"""Kaggriculture v3: diversified, staggered, market-aware baseline."""

import math

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS


BASE_PLAN = (
    [("MELON", 0)] * 16
    + [("TOMATO", 0)] * 9
)
EXTRA_PLAN = [("STRAWBERRY", 0)] * 5 + [("TOMATO", 0)] * 5
SELL_FLOORS = {
    "WHEAT": 18,
    "CARROT": 27,
    "TOMATO": 45,
    "STRAWBERRY": 90,
    "MELON": 180,
}
SELL_CAPS = {
    "WHEAT": 10,
    "CARROT": 8,
    "TOMATO": 8,
    "STRAWBERRY": 6,
    "MELON": 6,
}
SELL_ORDER = ("MELON", "STRAWBERRY", "TOMATO", "CARROT", "WHEAT")
LAND_COSTS = (1000, 2000, 4000)


def _pass(hand_count=0):
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in range(hand_count)],
        "market": [],
    }


def _quadrant(x, y, size):
    half = size // 2
    return ("S" if y >= half else "N") + ("E" if x >= half else "W")


def _ordered_tiles(farm, quadrant):
    size = len(farm.get("tiles", []))
    points = []
    for y, row in enumerate(farm.get("tiles", [])):
        for x, tile in enumerate(row):
            if tile != "LOCKED" and _quadrant(x, y, size) == quadrant:
                points.append((abs(x - 4) + abs(y - 4), y, x))
    points.sort()
    return [(x, y) for _, y, x in points]


def _crop_plan(farm):
    quadrants = list(farm.get("unlocked_quadrants", []) or ["NW"])
    plan = {}
    base = _ordered_tiles(farm, "NW")
    for pos, spec in zip(base, BASE_PLAN):
        plan[pos] = spec
    for quadrant in quadrants:
        if quadrant == "NW":
            continue
        for pos, spec in zip(_ordered_tiles(farm, quadrant), EXTRA_PLAN):
            plan[pos] = spec
    return plan


def _can_still_plant(crop, day):
    # Leave one full day after first maturity for harvesting and liquidation.
    return day + int(CROPS[crop]["first_yield_day"]) <= 28


def _task_list(obs, farm, private):
    day = int(obs.get("day", 0))
    seeds = dict(private.get("seeds", {}) or {})
    tasks = []
    for (x, y), (planned_crop, start_day) in _crop_plan(farm).items():
        tile = farm["tiles"][y][x]
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            actual_crop = tile.get("crop")
            crop_data = CROPS[actual_crop]
            age = day - int(tile.get("planted_day", day))
            harvest_age = (
                int(crop_data["first_yield_day"])
                if crop_data["ongoing"]
                else int(crop_data["max_yield_day"])
            )
            # One-time crops start with yield_units=1 before they are ripe.
            # Checking yield_units alone causes endless invalid HARVEST actions,
            # prevents watering, and turns the crop into a weed overnight.
            if age >= harvest_age and int(tile.get("yield_units", 0)) > 0:
                tasks.append((0, x, y, "HARVEST", None))
            elif not tile.get("watered_today", False):
                urgency = 1 if int(tile.get("consecutive_unwatered", 0)) else 2
                tasks.append((urgency, x, y, "WATER", None))
        elif isinstance(tile, dict) and tile.get("kind") == "WEED":
            tasks.append((1, x, y, "DIG", None))
        elif tile is None and day >= start_day and _can_still_plant(planned_crop, day):
            if int(seeds.get(planned_crop, 0)) > 0:
                tasks.append((3, x, y, "PLANT", planned_crop))
    return tasks


def _move(x, y, tx, ty):
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def _unit_actions(obs, farm, private):
    positions = [tuple(farm["farmer"])] + [tuple(p) for p in farm.get("hands", []) or []]
    tasks = _task_list(obs, farm, private)
    actions = [["PASS"] for _ in positions]
    remaining = dict(private.get("seeds", {}) or {})
    free_units = set(range(len(positions)))

    while free_units and tasks:
        best = None
        for unit_i in free_units:
            ux, uy = positions[unit_i]
            for task_i, task in enumerate(tasks):
                priority, tx, ty, op, crop = task
                if op == "PLANT" and int(remaining.get(crop, 0)) <= 0:
                    continue
                key = (priority, abs(ux - tx) + abs(uy - ty), ty, tx, unit_i)
                if best is None or key < best[0]:
                    best = (key, unit_i, task_i)
        if best is None:
            break
        _, unit_i, task_i = best
        _, tx, ty, op, crop = tasks.pop(task_i)
        ux, uy = positions[unit_i]
        if (ux, uy) == (tx, ty):
            actions[unit_i] = [op] if crop is None else [op, crop]
            if op == "PLANT":
                remaining[crop] = int(remaining.get(crop, 0)) - 1
        else:
            actions[unit_i] = _move(ux, uy, tx, ty)
        free_units.remove(unit_i)
    return actions


def _planned_work(obs, farm):
    day = int(obs.get("day", 0))
    work = 0
    for (x, y), (crop, start_day) in _crop_plan(farm).items():
        tile = farm["tiles"][y][x]
        if isinstance(tile, dict) and tile.get("kind") in ("PLANT", "WEED"):
            work += 1
        elif tile is None and day >= start_day and _can_still_plant(crop, day):
            work += 1
    return work


def _rival_crop_count(obs, crop):
    farms = obs.get("farms", []) or []
    player = int(obs.get("player", 0))
    if len(farms) != 2:
        return 0
    count = 0
    for row in farms[1 - player].get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == crop:
                count += 1
    return count


def _seed_needs(obs, farm, private):
    day = int(obs.get("day", 0))
    needs = {crop: 0 for crop in CROPS}
    for (x, y), (crop, start_day) in _crop_plan(farm).items():
        if farm["tiles"][y][x] is None and day >= start_day and _can_still_plant(crop, day):
            needs[crop] += 1
    seeds = private.get("seeds", {}) or {}
    return {crop: max(0, n - int(seeds.get(crop, 0))) for crop, n in needs.items()}


def _fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _market_actions(obs, farm, private):
    day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
    shed = private.get("shed", {}) or {}
    prices = ((obs.get("market", {}) or {}).get("prices", {}) or {})
    orders = []
    budget = float(farm.get("money", 0))

    # Sell on a six-times-per-day cadence so town demand can replenish prices.
    if hour % 4 == 0 or day >= 29:
        for item in SELL_ORDER:
            amount = int(shed.get(item, 0))
            if amount <= 0:
                continue
            price = int(prices.get(item, 0))
            floor = SELL_FLOORS[item]
            rival_wave = _rival_crop_count(obs, item) >= 8
            if rival_wave:
                floor = int(floor * 0.85)  # pre-empt an obvious rival harvest wave
            if day >= 28 or price >= floor:
                # If the rival visibly has a large crop wave, small throttled
                # sales lose: their bulk order crashes the shared price first.
                quantity = amount if day >= 29 or rival_wave else min(amount, SELL_CAPS[item])
                orders.append(["SELL", item, quantity])
                budget += quantity * max(1, price)

    quadrants = list(farm.get("unlocked_quadrants", []) or ["NW"])
    extra = max(0, len(quadrants) - 1)
    # One controlled expansion only, after the base farm has generated cash.
    if extra == 0 and 5 <= day <= 12 and budget >= 6000 and len(orders) < 10:
        orders.append(["BUY_LAND"])
        budget -= LAND_COSTS[0]

    for crop, need in _seed_needs(obs, farm, private).items():
        if need <= 0 or len(orders) >= 10:
            continue
        cost = int(CROPS[crop]["seed"])
        buy = min(need, max(0, int((budget - 250) // cost)))
        if buy > 0:
            orders.append(["BUY_SEED", crop, buy])
            budget -= buy * cost

    if hour <= 1:
        work = _planned_work(obs, farm)
        # Roughly ten useful tile operations per unit/day after travel overhead.
        desired_units = max(1, min(5, int(math.ceil(work / 10.0))))
        existing = 1 + len(farm.get("hands", []) or [])
        hire_index = int(farm.get("hires_today", 0))
        while existing < desired_units and len(orders) < 10:
            cost = _fib(hire_index)
            if budget < cost + 200:
                break
            orders.append(["HIRE"])
            budget -= cost
            hire_index += 1
            existing += 1
    return orders[:10]


def agent(obs):
    try:
        farms = obs.get("farms", []) or []
        player = int(obs.get("player", 0))
        if player < 0 or player >= len(farms):
            return _pass()
        farm = farms[player]
        private = obs.get("private", {}) or {}
        actions = _unit_actions(obs, farm, private)
        return {
            "farmer": actions[0] if actions else ["PASS"],
            "hands": actions[1:] if len(actions) > 1 else [],
            "market": _market_actions(obs, farm, private),
        }
    except Exception:
        try:
            farms = obs.get("farms", []) or []
            farm = farms[int(obs.get("player", 0))]
            return _pass(len(farm.get("hands", []) or []))
        except Exception:
            return _pass()
"""Overrides appended to melon_v3_agent.py to form the self-contained v4 agent."""

# A unit keeps its target coordinate until that tile no longer has work.
_V4_TARGETS = {}
_V4_LAST_STEP = -1
SAFE_PLANT_LAST_HOUR = 20
V4_BASE_PLAN = (
    [("MELON", 0)] * 13
    + [("MELON", 7)] * 3
    + [("TOMATO", 0)] * 8
    + [("TOMATO", 1)]
)


def _crop_plan(farm):
    quadrants = list(farm.get("unlocked_quadrants", []) or ["NW"])
    plan = {}
    for pos, spec in zip(_ordered_tiles(farm, "NW"), V4_BASE_PLAN):
        plan[pos] = spec
    for quadrant in quadrants:
        if quadrant == "NW":
            continue
        for pos, spec in zip(_ordered_tiles(farm, quadrant), EXTRA_PLAN):
            plan[pos] = spec
    return plan


def _can_still_plant(crop, day):
    crop_data = CROPS[crop]
    maturity = (
        int(crop_data["first_yield_day"])
        if crop_data["ongoing"]
        else int(crop_data["max_yield_day"])
    )
    # Mature by Day 27, leaving Days 28-29 for harvest and liquidation.
    return day + maturity <= 27


def _task_list(obs, farm, private):
    day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
    seeds = dict(private.get("seeds", {}) or {})
    tasks = []
    for (x, y), (planned_crop, start_day) in _crop_plan(farm).items():
        tile = farm["tiles"][y][x]
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            actual_crop = tile.get("crop")
            crop_data = CROPS[actual_crop]
            age = day - int(tile.get("planted_day", day))
            harvest_age = (
                int(crop_data["first_yield_day"])
                if crop_data["ongoing"]
                else int(crop_data["max_yield_day"])
            )
            if age >= harvest_age and int(tile.get("yield_units", 0)) > 0:
                tasks.append((0, x, y, "HARVEST", None))
            elif not tile.get("watered_today", False):
                urgency = 1 if int(tile.get("consecutive_unwatered", 0)) else 2
                tasks.append((urgency, x, y, "WATER", None))
        elif isinstance(tile, dict) and tile.get("kind") == "WEED":
            tasks.append((1, x, y, "DIG", None))
        elif (
            tile is None
            and hour <= SAFE_PLANT_LAST_HOUR
            and day >= start_day
            and _can_still_plant(planned_crop, day)
            and int(seeds.get(planned_crop, 0)) > 0
        ):
            tasks.append((3, x, y, "PLANT", planned_crop))
    return tasks


def _unit_actions(obs, farm, private):
    global _V4_TARGETS, _V4_LAST_STEP
    day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
    step = day * 24 + hour
    if step == 0 or step < _V4_LAST_STEP:
        _V4_TARGETS = {}
    _V4_LAST_STEP = step

    positions = [tuple(farm["farmer"])] + [tuple(p) for p in farm.get("hands", []) or []]
    tasks = _task_list(obs, farm, private)
    by_pos = {(t[1], t[2]): t for t in tasks}
    actions = [["PASS"] for _ in positions]
    remaining = dict(private.get("seeds", {}) or {})
    reserved = set()

    # Keep valid targets. This prevents workers swapping destinations each Turn.
    for unit_i in range(len(positions)):
        target = _V4_TARGETS.get(unit_i)
        if target not in by_pos or target in reserved:
            _V4_TARGETS.pop(unit_i, None)
        else:
            reserved.add(target)

    for unit_i, (ux, uy) in enumerate(positions):
        target = _V4_TARGETS.get(unit_i)
        if target is None:
            candidates = [t for t in tasks if (t[1], t[2]) not in reserved]
            if not candidates:
                continue
            task = min(candidates, key=lambda t: (t[0], abs(ux-t[1]) + abs(uy-t[2]), t[2], t[1]))
            target = (task[1], task[2])
            _V4_TARGETS[unit_i] = target
            reserved.add(target)

        task = by_pos[target]
        _, tx, ty, op, crop = task
        if (ux, uy) != (tx, ty):
            actions[unit_i] = _move(ux, uy, tx, ty)
            continue
        if op == "PLANT":
            if int(remaining.get(crop, 0)) <= 0:
                _V4_TARGETS.pop(unit_i, None)
                continue
            remaining[crop] = int(remaining.get(crop, 0)) - 1
            actions[unit_i] = ["PLANT", crop]
        else:
            actions[unit_i] = [op]
        # The coordinate remains locked. On the next observation PLANT may turn
        # into urgent WATER and HARVEST may turn into WATER, both desirable.
    return actions


def _planned_work(obs, farm):
    day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
    operations = 0
    active_tiles = 0
    for (x, y), (crop, start_day) in _crop_plan(farm).items():
        tile = farm["tiles"][y][x]
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            active_tiles += 1
            operations += 1  # daily water
            data = CROPS[tile.get("crop")]
            age = day - int(tile.get("planted_day", day))
            maturity = int(data["first_yield_day"] if data["ongoing"] else data["max_yield_day"])
            if age >= maturity and int(tile.get("yield_units", 0)) > 0:
                operations += 1  # harvest as well as water/replant work
        elif isinstance(tile, dict) and tile.get("kind") == "WEED":
            active_tiles += 1
            operations += 3  # dig, plant, water
        elif tile is None and hour <= SAFE_PLANT_LAST_HOUR and day >= start_day and _can_still_plant(crop, day):
            active_tiles += 1
            operations += 2  # plant and water

    # Approximate one move between neighboring jobs. Twenty of 24 turns are
    # treated as safely usable so a late spawn or awkward route has slack.
    return operations + active_tiles


def _market_actions(obs, farm, private):
    # Reuse v3 market logic, but `_planned_work`, `_seed_needs`, and
    # `_can_still_plant` resolve to the v4 overrides in this combined module.
    day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
    shed = private.get("shed", {}) or {}
    prices = ((obs.get("market", {}) or {}).get("prices", {}) or {})
    orders, budget = [], float(farm.get("money", 0))

    if hour % 4 == 0 or day >= 29:
        for item in SELL_ORDER:
            amount = int(shed.get(item, 0))
            if amount <= 0:
                continue
            price, floor = int(prices.get(item, 0)), SELL_FLOORS[item]
            if day >= 28 or price >= floor:
                # A visible crop count is not a reliable harvest forecast. Sell
                # the available batch while its observed quote clears the floor.
                quantity = amount
                orders.append(["SELL", item, quantity])
                budget += quantity * max(1, price)

    quadrants = list(farm.get("unlocked_quadrants", []) or ["NW"])
    if len(quadrants) == 1 and 5 <= day <= 12 and budget >= 6000 and len(orders) < 10:
        orders.append(["BUY_LAND"])
        budget -= LAND_COSTS[0]

    for crop, need in _seed_needs(obs, farm, private).items():
        if need <= 0 or len(orders) >= 10:
            continue
        cost = int(CROPS[crop]["seed"])
        buy = min(need, max(0, int((budget - 250) // cost)))
        if buy > 0:
            orders.append(["BUY_SEED", crop, buy])
            budget -= buy * cost

    if hour <= 1:
        estimated_turns = _planned_work(obs, farm)
        desired_units = max(1, min(6, int(math.ceil(estimated_turns / 20.0))))
        existing = 1 + len(farm.get("hands", []) or [])
        hire_index = int(farm.get("hires_today", 0))
        while existing < desired_units and len(orders) < 10:
            cost = _fib(hire_index)
            if budget < cost + 200:
                break
            orders.append(["HIRE"])
            budget -= cost
            hire_index += 1
            existing += 1
    return orders[:10]
"""Animal- and shop-aware overrides appended after the v4 agent."""

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, SHOPS

_V5_TARGETS = {}
_V5_LAST_STEP = -1
V5_ANIMAL_SLOTS = 10
V5_BASE_PLAN = [("MELON", 0)] * 10 + [("MELON", 7)] * 2 + [("TOMATO", 0)] * 13
V5_SELL_FLOORS = dict(SELL_FLOORS, MILK=115, WOOL=145, EGG=38)
V5_SELL_ORDER = ("WOOL", "MILK", "MELON", "STRAWBERRY", "TOMATO", "CARROT", "WHEAT", "EGG")


def _crop_plan(farm):
    # Keep the proven crop core in NW. Expanded NE is reserved for animals so
    # crop and structure tasks can never compete for the same coordinate.
    plan = {}
    for pos, spec in zip(_ordered_tiles(farm, "NW"), V5_BASE_PLAN):
        plan[pos] = spec
    return plan


def _shop_pull(obs, item):
    shops = ((obs.get("town", {}) or {}).get("unlocked_shops", []) or [])
    return sum(item in SHOPS.get(shop, []) for shop in shops)


def _shed_access(size):
    half = size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def _animal_positions(farm):
    quadrants = list(farm.get("unlocked_quadrants", []) or ["NW"])
    if "NE" not in quadrants:
        return []
    blocked = set(_shed_access(len(farm.get("tiles", []))))
    return [p for p in _ordered_tiles(farm, "NE") if p not in blocked][:V5_ANIMAL_SLOTS]


def _animal_targets(obs, farm):
    slots = _animal_positions(farm)
    milk_pull = _shop_pull(obs, "MILK")
    wool_pull = _shop_pull(obs, "WOOL")
    milk_price = int((((obs.get("market", {}) or {}).get("prices", {}) or {}).get("MILK", 160)))
    wool_price = int((((obs.get("market", {}) or {}).get("prices", {}) or {}).get("WOOL", 200)))
    milk_score = (1 + milk_pull) * milk_price / 160.0
    wool_score = (1 + wool_pull) * wool_price / 200.0
    if milk_score >= wool_score * 1.35:
        cows = 7
    elif wool_score >= milk_score * 1.35:
        cows = 3
    else:
        cows = 5
    return {pos: ("COW" if i < cows else "SHEEP") for i, pos in enumerate(slots)}


def _crop_tasks(obs, farm, private):
    day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
    seeds = dict(private.get("seeds", {}) or {})
    tasks = []
    for (x, y), (planned_crop, start_day) in _crop_plan(farm).items():
        tile = farm["tiles"][y][x]
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop = tile.get("crop")
            data = CROPS[crop]
            age = day - int(tile.get("planted_day", day))
            maturity = int(data["first_yield_day"] if data["ongoing"] else data["max_yield_day"])
            if age >= maturity and int(tile.get("yield_units", 0)) > 0:
                tasks.append((1, x, y, "HARVEST", None))
            elif not tile.get("watered_today", False):
                priority = 0 if int(tile.get("consecutive_unwatered", 0)) else 2
                tasks.append((priority, x, y, "WATER", None))
        elif isinstance(tile, dict) and tile.get("kind") == "WEED":
            tasks.append((0, x, y, "DIG", None))
        elif (
            tile is None and hour <= SAFE_PLANT_LAST_HOUR and day >= start_day
            and _can_still_plant(planned_crop, day) and int(seeds.get(planned_crop, 0)) > 0
        ):
            tasks.append((4, x, y, "PLANT", planned_crop))
    return tasks


def _animal_tasks(obs, farm):
    tasks = []
    for (x, y), wanted in _animal_targets(obs, farm).items():
        tile = farm["tiles"][y][x]
        if tile is None:
            tasks.append((3, x, y, "BUILD_PASTURE", None))
        elif isinstance(tile, dict) and tile.get("kind") == "WEED":
            tasks.append((0, x, y, "DIG", None))
        elif isinstance(tile, dict) and tile.get("kind") == "PASTURE" and "animal" not in tile:
            tasks.append((2, x, y, "PLACE", wanted))
        elif isinstance(tile, dict) and "animal" in tile:
            if not tile.get("fed_today", False):
                tasks.append((0, x, y, "FEED", "WHEAT"))
            elif int(tile.get("yield_units", 0)) > 0:
                tasks.append((1, x, y, "HARVEST", None))
            elif not tile.get("cared_today", False):
                tasks.append((2, x, y, "CARE", None))
    return tasks


def _task_list(obs, farm, private):
    return _crop_tasks(obs, farm, private) + _animal_tasks(obs, farm)


def _unit_actions(obs, farm, private):
    global _V5_TARGETS, _V5_LAST_STEP
    day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
    step = day * 24 + hour
    if step == 0 or hour == 0 or step < _V5_LAST_STEP:
        _V5_TARGETS = {}
    _V5_LAST_STEP = step

    positions = [tuple(farm["farmer"])] + [tuple(p) for p in farm.get("hands", []) or []]
    inventories = list(private.get("inventories", []) or [])
    while len(inventories) < len(positions):
        inventories.append({})
    tasks = _task_list(obs, farm, private)
    by_pos = {(t[1], t[2]): t for t in tasks}
    actions = [["PASS"] for _ in positions]
    reserved = set()
    remaining_seeds = dict(private.get("seeds", {}) or {})
    available_shed = dict(private.get("shed", {}) or {})

    for unit_i in range(len(positions)):
        target = _V5_TARGETS.get(unit_i)
        if target not in by_pos or target in reserved:
            _V5_TARGETS.pop(unit_i, None)
        else:
            reserved.add(target)

    for unit_i, (ux, uy) in enumerate(positions):
        inv = inventories[unit_i] or {}
        target = _V5_TARGETS.get(unit_i)
        if target is None:
            candidates = []
            for task in tasks:
                pos = (task[1], task[2])
                if pos in reserved:
                    continue
                need = task[4] if task[3] in ("FEED", "PLACE") else None
                if need and int(inv.get(need, 0)) <= 0 and int(available_shed.get(need, 0)) <= 0:
                    continue
                candidates.append(task)
            if not candidates:
                continue
            task = min(candidates, key=lambda t: (t[0], abs(ux-t[1]) + abs(uy-t[2]), t[2], t[1]))
            target = (task[1], task[2])
            _V5_TARGETS[unit_i] = target
            reserved.add(target)

        task = by_pos[target]
        _, tx, ty, op, arg = task
        required = arg if op in ("FEED", "PLACE") else None
        if required and int(inv.get(required, 0)) <= 0:
            if int(available_shed.get(required, 0)) <= 0:
                _V5_TARGETS.pop(unit_i, None)
                continue
            access = min(_shed_access(len(farm["tiles"])), key=lambda p: abs(ux-p[0]) + abs(uy-p[1]))
            if (ux, uy) == access:
                actions[unit_i] = ["PICKUP", required, 1]
                available_shed[required] = int(available_shed.get(required, 0)) - 1
            else:
                actions[unit_i] = _move(ux, uy, access[0], access[1])
            continue

        if (ux, uy) != (tx, ty):
            actions[unit_i] = _move(ux, uy, tx, ty)
            continue
        if op == "PLANT":
            if int(remaining_seeds.get(arg, 0)) <= 0:
                _V5_TARGETS.pop(unit_i, None)
                continue
            remaining_seeds[arg] = int(remaining_seeds.get(arg, 0)) - 1
            actions[unit_i] = ["PLANT", arg]
        elif op == "PLACE":
            actions[unit_i] = ["PLACE", arg]
        else:
            actions[unit_i] = [op]
    return actions


def _total_private(private, item):
    total = int((private.get("shed", {}) or {}).get(item, 0))
    for inv in private.get("inventories", []) or []:
        total += int((inv or {}).get(item, 0))
    return total


def _placed_animals(farm):
    out = {"COW": 0, "SHEEP": 0}
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") in out:
                out[tile["animal"]] += 1
    return out


def _planned_work(obs, farm):
    private = obs.get("private", {}) or {}
    tasks = _task_list(obs, farm, private)
    # FEED/PLACE usually require a shed trip plus a cross-quadrant walk. Treat
    # them as ten turns, not merely one tile operation.
    return sum(10 if t[3] in ("FEED", "PLACE") else 2 for t in tasks)


def _market_actions(obs, farm, private):
    day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
    shed = private.get("shed", {}) or {}
    prices = ((obs.get("market", {}) or {}).get("prices", {}) or {})
    orders, budget = [], float(farm.get("money", 0))

    if hour % 4 == 0 or day >= 29:
        for item in V5_SELL_ORDER:
            amount = int(shed.get(item, 0))
            if item == "WHEAT" and day < 28:
                amount = max(0, amount - 30)
            if amount <= 0:
                continue
            floor = V5_SELL_FLOORS.get(item, 1)
            if day >= 28 or int(prices.get(item, 0)) >= floor:
                orders.append(["SELL", item, amount])
                budget += amount * max(1, int(prices.get(item, 1)))

    quadrants = list(farm.get("unlocked_quadrants", []) or ["NW"])
    if len(quadrants) == 1 and 12 <= day <= 17 and budget >= 6000 and len(orders) < 10:
        orders.append(["BUY_LAND"])
        budget -= 1000

    if "NE" in quadrants:
        targets = _animal_targets(obs, farm)
        wanted = {"COW": sum(a == "COW" for a in targets.values()), "SHEEP": sum(a == "SHEEP" for a in targets.values())}
        placed = _placed_animals(farm)
        for animal in ("COW", "SHEEP"):
            owned = placed[animal] + _total_private(private, animal)
            missing = max(0, wanted[animal] - owned)
            cost = int(ANIMALS[animal]["cost"])
            buy = min(missing, 2, max(0, int((budget - 900) // cost)))
            if buy > 0 and len(orders) < 10:
                orders.append(["BUY_ANIMAL", animal, buy])
                budget -= buy * cost

        animal_count = sum(placed.values())
        wheat_needed = max(0, animal_count * 3 - _total_private(private, "WHEAT"))
        if wheat_needed > 0 and len(orders) < 10:
            wheat_price = max(1, int(prices.get("WHEAT", 25)))
            buy = min(wheat_needed, max(0, int((budget - 500) // wheat_price)))
            if buy > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", buy])
                budget -= buy * wheat_price

    for crop, need in _seed_needs(obs, farm, private).items():
        if need <= 0 or len(orders) >= 10:
            continue
        cost = int(CROPS[crop]["seed"])
        buy = min(need, max(0, int((budget - 350) // cost)))
        if buy > 0:
            orders.append(["BUY_SEED", crop, buy])
            budget -= buy * cost

    if hour <= 1:
        estimated = _planned_work(obs, farm)
        desired_units = max(1, min(10, int(math.ceil(estimated / 16.0))))
        existing = 1 + len(farm.get("hands", []) or [])
        hire_index = int(farm.get("hires_today", 0))
        hire_orders = []
        hire_budget = float(farm.get("money", 0))
        while existing < desired_units and len(hire_orders) < 9:
            cost = _fib(hire_index)
            if hire_budget < cost + 300:
                break
            hire_orders.append(["HIRE"])
            hire_budget -= cost
            hire_index += 1
            existing += 1
        # Labor is the daily hard constraint: place HIRE before purchases and
        # let low-priority seed/animal orders spill into the next Turn.
        orders = hire_orders + orders
    return orders[:10]


opening = tuple([("MELON", 0)] * 10 + [("WHEAT", 0)] * 9)
expansion_crops = tuple([("STRAWBERRY", 6)] * 50)
expansion_crop_policy = None
expansion_crop_limit_policy = None
animals = tuple(["COW"] * 6)
animal_capacity = 6
animal_start_day = 0
animal_policy = True
product_keeps = {"MILK": 5, "WOOL": 5, "EGG": 5}
liquidation_day = 6
land_target = None
reserve = 500
land_reserve = 3000
feed_reserve_units = None
execution_fix = True
wheat_daily_sales = True
land_min_plants = 0
land_setup_task_limit = None
feed_batch = 2
opportunistic_fertilize_policy = None
labor_work_divisor = 16
labor_min_by_land = {}
build_executor = None
_FINANCING_STATE = {"last_step": -1, "threshold": None}
ANIMAL_DELAY_STEP = 72


def _financing_threshold(obs):
    day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
    step = day * 24 + hour
    if step == 0 or step < _FINANCING_STATE["last_step"]:
        _FINANCING_STATE.update(last_step=-1, threshold=None)
    _FINANCING_STATE["last_step"] = step
    if ANIMAL_DELAY_STEP is not None:
        return ANIMAL_DELAY_STEP
    if _FINANCING_STATE["threshold"] is None and step >= 4:
        farms = obs.get("farms", []) or []
        player = int(obs.get("player", 0))
        rival = farms[1 - player] if len(farms) == 2 else {}
        rival_units = 1 + len(rival.get("hands", []) or [])
        _FINANCING_STATE["threshold"] = 4 if rival_units > 5 else 24
    return _FINANCING_STATE["threshold"] if _FINANCING_STATE["threshold"] is not None else 24


def desired_land_target(_obs):
    return 3


def desired_animals(obs):
    day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
    step = day * 24 + hour
    return ["COW"] * (4 if step < _financing_threshold(obs) else 6)


def expansion_batch(_farm):
    return 0


def labor_floor_policy(_obs, land_count):
    return {2: 9, 3: 11}.get(land_count, 0)


def labor_cap(_obs):
    return 12


def melon_liquidation_policy(_obs):
    return False


land_target = desired_land_target
def _crop_plan(farm):
    plan = dict(zip(_ordered_tiles(farm, "NW"), opening))
    available_expansion_crops = (
        tuple(expansion_crop_policy(farm))
        if callable(expansion_crop_policy)
        else expansion_crops
    )
    if callable(expansion_crop_limit_policy):
        limit = max(0, int(expansion_crop_limit_policy(farm)))
        available_expansion_crops = expansion_crops[:limit]
    offset = 0
    unlocked = list(farm.get("unlocked_quadrants", []) or ["NW"])
    for quadrant in ("NE", "SW", "SE"):
        if quadrant not in unlocked:
            continue
        points = _ordered_tiles(farm, quadrant)
        specs = available_expansion_crops[offset:offset + len(points)]
        batch = expansion_batch(farm) if callable(expansion_batch) else expansion_batch
        if batch:
            planted = sum(
                isinstance(farm["tiles"][y][x], dict)
                and farm["tiles"][y][x].get("kind") == "PLANT"
                for x, y in points
            )
            allowed = min(len(points), int(batch))
            while allowed < len(points) and planted >= max(1, allowed - 2):
                allowed = min(len(points), allowed + int(batch))
            specs = specs[:allowed]
        plan.update(zip(points, specs))
        offset += len(points)
    return plan


def _animal_positions(farm):
    blocked = set(_shed_access(len(farm.get("tiles", []))))
    blocked.update(_crop_plan(farm))
    positions = []
    unlocked = list(farm.get("unlocked_quadrants", []) or ["NW"])
    for quadrant in ("NW", "NE", "SW", "SE"):
        if quadrant in unlocked:
            positions.extend(
                p for p in _ordered_tiles(farm, quadrant)
                if p not in blocked
            )
    return positions[:animal_capacity]


def _animal_targets(obs, farm):
    if int(obs.get("day", 0)) < animal_start_day:
        return {}
    return dict(zip(_animal_positions(farm), desired_animals(obs)))


def _animal_tasks(obs, farm):
    tasks = []
    targets = _animal_targets(obs, farm)
    for (x, y), wanted in targets.items():
        tile = farm["tiles"][y][x]
        if tile is None:
            structure = ANIMALS[wanted]["structure"]
            tasks.append((3, x, y, f"BUILD_{structure}", None))
        elif isinstance(tile, dict) and tile.get("kind") == "WEED":
            tasks.append((0, x, y, "DIG", None))
        elif (isinstance(tile, dict)
              and tile.get("kind") == ANIMALS[wanted]["structure"]
              and "animal" not in tile):
            tasks.append((-1 if execution_fix else 2, x, y, "PLACE", wanted))
        elif isinstance(tile, dict) and "animal" in tile:
            if not tile.get("fed_today", False) and int(obs.get("day", 0)) < 29:
                urgent = int(tile.get("consecutive_unfed", 0)) >= 1
                tasks.append(((-2 if urgent else -1) if execution_fix else 0,
                              x, y, "FEED", "WHEAT"))
            elif int(tile.get("yield_units", 0)) > 0:
                tasks.append((0 if execution_fix else 1, x, y, "HARVEST", None))
            elif tile.get("fertilizer_available", False):
                tasks.append((2, x, y, "COLLECT_FERTILIZER", None))
            elif not tile.get("cared_today", False) and int(obs.get("day", 0)) < 29:
                tasks.append((3, x, y, "CARE", None))
    return tasks


def _unit_actions(obs, farm, private):
    global _V5_TARGETS, _V5_LAST_STEP
    day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
    step = day * 24 + hour
    if step == 0 or hour == 0 or step < _V5_LAST_STEP:
        _V5_TARGETS = {}
    _V5_LAST_STEP = step

    positions = [tuple(farm["farmer"])] + [tuple(p) for p in farm.get("hands", []) or []]
    inventories = list(private.get("inventories", []) or [])
    while len(inventories) < len(positions):
        inventories.append({})
    tasks = _task_list(obs, farm, private)
    by_pos = {(task[1], task[2]): task for task in tasks}
    actions = [["PASS"] for _ in positions]
    reserved = set()
    available_shed = dict(private.get("shed", {}) or {})

    for unit_i in range(len(positions)):
        target = _V5_TARGETS.get(unit_i)
        if target not in by_pos or target in reserved:
            _V5_TARGETS.pop(unit_i, None)
        else:
            reserved.add(target)

    for unit_i, (ux, uy) in enumerate(positions):
        inv = inventories[unit_i] or {}
        if callable(drop_policy) and inv and drop_policy(obs, farm, private, unit_i, inv):
            _V5_TARGETS.pop(unit_i, None)
            access = min(_shed_access(len(farm["tiles"])),
                         key=lambda p: abs(ux-p[0]) + abs(uy-p[1]))
            actions[unit_i] = (["DROP"] if (ux, uy) == access
                               else _move(ux, uy, access[0], access[1]))
            continue
        target = _V5_TARGETS.get(unit_i)
        if target is None:
            candidates = []
            normal_distances = [
                abs(ux-task[1]) + abs(uy-task[2])
                for task in tasks
                if task[3] != "FERTILIZE" and (task[1], task[2]) not in reserved
            ]
            normal_distance = min(normal_distances) if normal_distances else None
            for task in tasks:
                pos = (task[1], task[2])
                if pos in reserved:
                    continue
                distance = abs(ux-task[1]) + abs(uy-task[2])
                if task[3] == "FERTILIZE":
                    if fertilizer_carry_only and int(inv.get("FERTILIZER", 0)) <= 0:
                        continue
                    if (fertilizer_max_distance is not None
                            and distance > int(fertilizer_max_distance)):
                        continue
                    if (fertilizer_max_detour is not None and normal_distance is not None
                            and distance > normal_distance + int(fertilizer_max_detour)):
                        continue
                need = task[4] if task[3] in ("FEED", "PLACE") else None
                if need and int(inv.get(need, 0)) <= 0 and int(available_shed.get(need, 0)) <= 0:
                    continue
                candidates.append(task)
            if not candidates:
                continue
            task = min(candidates, key=lambda t: (t[0], abs(ux-t[1]) + abs(uy-t[2]), t[2], t[1]))
            target = (task[1], task[2])
            _V5_TARGETS[unit_i] = target
            reserved.add(target)

        task = by_pos[target]
        _, tx, ty, op, arg = task
        required = arg if op in ("FEED", "PLACE", "FERTILIZE") else None
        if required and int(inv.get(required, 0)) <= 0:
            available = int(available_shed.get(required, 0))
            if available <= 0:
                _V5_TARGETS.pop(unit_i, None)
                continue
            access = min(_shed_access(len(farm["tiles"])), key=lambda p: abs(ux-p[0]) + abs(uy-p[1]))
            if (ux, uy) == access:
                quantity = min(available, (feed_batch if required == "WHEAT"
                                           else fertilizer_pickup_batch if required == "FERTILIZER"
                                           else 1))
                actions[unit_i] = ["PICKUP", required, quantity]
                available_shed[required] -= quantity
            else:
                actions[unit_i] = _move(ux, uy, access[0], access[1])
            continue
        if (ux, uy) != (tx, ty):
            actions[unit_i] = _move(ux, uy, tx, ty)
        elif op == "PLACE":
            actions[unit_i] = ["PLACE", arg]
        elif op == "PLANT":
            actions[unit_i] = ["PLANT", arg]
        else:
            actions[unit_i] = [op]
    if callable(opportunistic_fertilize_policy):
        for unit_i, (ux, uy) in enumerate(positions):
            inv = inventories[unit_i] or {}
            if (actions[unit_i] == ["PASS"]
                    and int(inv.get("FERTILIZER", 0)) > 0
                    and opportunistic_fertilize_policy(
                        obs, farm, private, unit_i, (ux, uy), inv)):
                actions[unit_i] = ["FERTILIZE"]
    return actions


def _base_market_actions(obs, farm, private):
    day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
    shed = private.get("shed", {}) or {}
    prices = ((obs.get("market", {}) or {}).get("prices", {}) or {})
    orders, budget = [], float(farm.get("money", 0))

    desired = desired_animals(obs)
    possible_animals = set(animals) | set(desired)
    if animal_policy is not None:
        possible_animals.update(("COW", "SHEEP"))
    products = {ANIMALS[a]["product"] for a in possible_animals}
    for item in ("FERTILIZER", *sorted(products)):
        amount = int(shed.get(item, 0))
        keep = product_keeps.get(item, 0)
        keep = int(keep(obs) if callable(keep) else keep)
        sell = amount if day >= 29 else max(0, amount - keep)
        if sell:
            orders.append(["SELL", item, sell])
            budget += sell * max(1, int(prices.get(item, 1)))

    # Buying and selling use the same net feed target.  Account for wheat
    # carried by workers so a shed-only threshold cannot sell feed that the
    # next market calculation immediately buys back.
    feed_reserve = int(feed_reserve_units if feed_reserve_units is not None
                       else len(desired) * 4)
    planned_wheat_sale = 0
    for item in ("MELON", "TOMATO", "WHEAT", "STRAWBERRY"):
        amount = int(shed.get(item, 0))
        if item == "WHEAT" and day < 29:
            if wheat_daily_sales and hour > 1:
                amount = 0
            carried = max(0, _total_private(private, "WHEAT") - amount)
            protected_shed = max(0, feed_reserve - carried)
            amount = max(0, amount - protected_shed)
        if amount <= 0:
            continue
        floor = {"MELON": 275, "TOMATO": 55, "WHEAT": 24, "STRAWBERRY": 110}[item]
        target_land = desired_land_target(obs)
        financing_wave = (
            item == "MELON"
            and day >= liquidation_day
            and len(farm.get("unlocked_quadrants", []) or []) < target_land
        )
        if item == "MELON" and melon_liquidation_policy is not None:
            financing_wave = financing_wave or bool(melon_liquidation_policy(obs))
        if day >= 29 or financing_wave or int(prices.get(item, 0)) >= floor:
            orders.append(["SELL", item, amount])
            budget += amount * max(1, int(prices.get(item, 1)))
            if item == "WHEAT":
                planned_wheat_sale = amount

    unlocked = len(farm.get("unlocked_quadrants", []) or ["NW"])
    target_land = desired_land_target(obs)
    while day >= liquidation_day and unlocked < target_land and len(orders) < 10:
        if land_setup_task_limit is not None:
            setup_ops = {"DIG", "PLANT", "BUILD_PASTURE", "PLACE"}
            pending_setup = sum(
                task[3] in setup_ops for task in _task_list(obs, farm, private)
            )
            if pending_setup > int(land_setup_task_limit):
                break
        if land_min_plants and unlocked >= 2:
            prior_quadrant = ("NE", "SW", "SE")[unlocked - 2]
            prior_points = _ordered_tiles(farm, prior_quadrant)
            prior_plants = sum(
                isinstance(farm["tiles"][y][x], dict)
                and farm["tiles"][y][x].get("kind") == "PLANT"
                for x, y in prior_points
            )
            if prior_plants < int(land_min_plants):
                break
        cost = LAND_COSTS[min(unlocked - 1, len(LAND_COSTS) - 1)]
        if budget < cost + land_reserve:
            break
        orders.append(["BUY_LAND"])
        budget -= cost
        unlocked += 1

    placed = {species: 0 for species in ANIMALS}
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") in placed:
                placed[tile["animal"]] += 1
    for species in ("COW", "SHEEP", "GOOSE") if day >= animal_start_day else ():
        owned = int(placed[species]) + _total_private(private, species)
        missing = max(0, desired.count(species) - owned)
        cost = int(ANIMALS[species]["cost"])
        buy = min(missing, 2, max(0, int((budget - reserve) // cost)))
        if buy:
            orders.append(["BUY_ANIMAL", species, buy])
            budget -= buy * cost

    wheat_total = max(0, _total_private(private, "WHEAT") - planned_wheat_sale)
    wheat_target = 0 if day >= 29 else feed_reserve
    need = max(0, wheat_target - wheat_total)
    price = max(1, int(prices.get("WHEAT", 25)))
    buy = min(need, max(0, int((budget - reserve) // price)))
    if buy:
        orders.append(["BUY_PRODUCT", "WHEAT", buy])
        budget -= buy * price

    for crop, need in _seed_needs(obs, farm, private).items():
        cost = int(CROPS[crop]["seed"])
        buy = min(need, max(0, int((budget - reserve) // cost)))
        if buy:
            orders.append(["BUY_SEED", crop, buy])
            budget -= buy * cost

    if hour <= 1:
        tasks = _task_list(obs, farm, private)
        cap = labor_cap(obs) if callable(labor_cap) else labor_cap
        if execution_fix:
            weights = {"FEED": 8, "PLACE": 8, "PLANT": 3,
                       "HARVEST": 2, "WATER": 2, "DIG": 2,
                       "COLLECT_FERTILIZER": 2, "CARE": 2}
            work = sum(weights.get(task[3], 2) for task in tasks)
            desired = max(2, min(int(cap), int(math.ceil(work / float(labor_work_divisor)))))
        else:
            work = len(tasks)
            desired = max(2, min(int(cap), int(math.ceil(work / 7.0))))
        land_count = len(farm.get("unlocked_quadrants", []) or ["NW"])
        floor = (labor_floor_policy(obs, land_count)
                 if callable(labor_floor_policy)
                 else labor_min_by_land.get(land_count, 0))
        desired = max(desired, int(floor))
        existing = 1 + len(farm.get("hands", []) or [])
        hire_i = int(farm.get("hires_today", 0))
        hires = []
        while existing < desired and len(hires) + len(orders) < 10:
            cost = _fib(hire_i)
            if budget < cost + reserve:
                break
            hires.append(["HIRE"])
            budget -= cost
            hire_i += 1
            existing += 1
        orders = hires + orders
    # Optional local-evaluation hook.  It observes the full queue before
    # the official 10-order cap without changing submitted behavior.
    diagnostic_hook = getattr(build_executor, "market_diagnostic_hook", None)
    if callable(diagnostic_hook):
        diagnostic_hook(obs, farm, private, list(orders))
    return orders[:10]


def drop_policy(obs, _farm, _private, _unit_i, inv):
    if int(obs.get("day", 0)) < 29:
        return False
    return any(int(inv.get(item, 0)) > 0 for item in
               ("MELON", "STRAWBERRY", "MILK", "WOOL", "EGG", "FERTILIZER", "WHEAT"))


_base_task_list = _task_list


def _task_list(obs, farm, private):
    tasks = _base_task_list(obs, farm, private)
    if int(obs.get("day", 0)) < 26:
        return tasks
    farms = obs.get("farms", []) or []
    player = int(obs.get("player", 0))
    rival = farms[1 - player] if len(farms) == 2 else {}
    animals = {"COW": 0, "SHEEP": 0, "GOOSE": 0}
    rival_strawberries = 0
    for row in rival.get("tiles", []) or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            species = tile.get("animal")
            if species in animals:
                animals[species] += 1
            if tile.get("kind") == "PLANT" and tile.get("crop") == "STRAWBERRY":
                rival_strawberries += 1
    if (sum(animals.values()) < 6 or rival_strawberries > 0
            or sum(count > 0 for count in animals.values()) != 1):
        return tasks
    adjusted = []
    for priority, x, y, op, arg in tasks:
        tile = farm["tiles"][y][x]
        if (op == "HARVEST" and isinstance(tile, dict)
                and tile.get("crop") == "STRAWBERRY"):
            priority = -2
        adjusted.append((priority, x, y, op, arg))
    return adjusted


_R10_STATE = {"last_step": -1}


def _rival_ready_melons(obs):
    farms = obs.get("farms", []) or []
    player = int(obs.get("player", 0))
    rival = farms[1 - player] if len(farms) == 2 else {}
    return sum(
        int(tile.get("yield_units", 0))
        for row in rival.get("tiles", []) or [] for tile in row
        if isinstance(tile, dict) and tile.get("kind") == "PLANT"
        and tile.get("crop") == "MELON"
    )


def _market_actions(obs, farm, private):
    step = int(obs.get("day", 0)) * 24 + int(obs.get("hour", 0))
    if step == 0 or step < _R10_STATE["last_step"]:
        _R10_STATE["last_step"] = -1
    _R10_STATE["last_step"] = step
    orders = list(_base_market_actions(obs, farm, private))
    has_melon_sale = any(
        len(order) >= 3 and order[0] == "SELL" and order[1] == "MELON"
        for order in orders
    )
    if (not has_melon_sale and int(obs.get("day", 0)) < 29
            and _rival_ready_melons(obs) > 0 and len(orders) < 10):
        amount = min(int((private.get("shed") or {}).get("MELON", 0)), 10)
        if amount > 0:
            orders.append(["SELL", "MELON", amount])
    return orders[:10]


_capacity_original_market = _market_actions
def _market_actions(obs, farm, private):
    orders = _capacity_original_market(obs, farm, private)
    if int(obs.get('hour', 0)) < 20 or int(obs.get('day', 0)) >= 29:
        return orders
    shed = dict(private.get('shed') or {})
    total = sum(shed.values()) + sum(sum(inv.values()) for inv in
                                   (private.get('inventories') or []) if inv)
    # Respect already-planned sales and purchases; retain ten free
    # slots for the remaining late-day harvest (heuristic to calibrate).
    projected = total
    for order in orders:
        if len(order) >= 3 and order[0] == 'SELL':
            projected -= min(int(shed.get(order[1], 0)), int(order[2]))
        elif len(order) >= 3 and order[0] in ('BUY_PRODUCT', 'BUY_ANIMAL'):
            projected += int(order[2])
    need = max(0, projected - 90)
    extra=[]
    already_sold={o[1] for o in orders if len(o)>=3 and o[0]=='SELL'}
    # First test the blocked crop stock, preserving animal feed. This
    # is a capacity repair, not a claim of optimal inventory valuation.
    for product in ('MELON', 'STRAWBERRY', 'TOMATO'):
        if product in already_sold or need <= 0: continue
        amount=min(need, int(shed.get(product,0)))
        if amount:
            extra.append(['SELL', product, amount]); need-=amount
    return (extra + orders)[:10]


def submission_agent(obs):
    try:
        farms = obs.get("farms", []) or []
        player = int(obs.get("player", 0))
        if player < 0 or player >= len(farms):
            return _pass()
        farm = farms[player]
        private = obs.get("private", {}) or {}
        actions = _unit_actions(obs, farm, private)
        return {"farmer": actions[0] if actions else ["PASS"],
                "hands": actions[1:] if len(actions) > 1 else [],
                "market": _market_actions(obs, farm, private)}
    except Exception:
        try:
            farm = (obs.get("farms", []) or [])[int(obs.get("player", 0))]
            return _pass(len(farm.get("hands", []) or []))
        except Exception:
            return _pass()
