import random
from typing import Dict, List, Tuple, Any, Optional

def _manhattan_dist(x1: int, y1: int, x2: int, y2: int) -> int:
    """Calculates Manhattan distance between two grid cells."""
    return abs(x1 - x2) + abs(y1 - y2)

def _get_path_step(fx: int, fy: int, tx: int, ty: int, reserved_positions: set, board_size: int) -> str:
    """
    Calculates an optimal movement vector toward a target position
    while actively avoiding spatial collision with other friendly workers.
    """
    directions = []
    if fx > tx: directions.append(("WEST", fx - 1, fy))
    if fx < tx: directions.append(("EAST", fx + 1, fy))
    if fy > ty: directions.append(("NORTH", fx, fy - 1))
    if fy < ty: directions.append(("SOUTH", fx, fy + 1))
    
    # 1. Try to take the primary, direct steps toward the goal
    for move, nx, ny in directions:
        if 0 <= nx < board_size and 0 <= ny < board_size:
            if (nx, ny) not in reserved_positions:
                reserved_positions.add((nx, ny))
                return move
                
    # 2. Path Obstruction Fallback: Attempt alternative open axes if preferred route is blocked
    for move, dx, dy in [("EAST", 1, 0), ("WEST", -1, 0), ("SOUTH", 0, 1), ("NORTH", 0, -1)]:
        nx, ny = fx + dx, fy + dy
        if 0 <= nx < board_size and 0 <= ny < board_size and (nx, ny) not in reserved_positions:
            reserved_positions.add((nx, ny))
            return move
            
    return "PASS"

def _evaluate_crop_roi(market_prices: Dict[str, float]) -> str:
    """
    Calculates the true daily compounding margin of crops based on 
    current fluctuating market values and standard growth timelines.
    """
    best_crop = "WHEAT"
    max_roi_per_day = -999.0
    
    # Robust static specifications to defend against dynamic global schema failures
    crop_specs = {
        "WHEAT": {"seed": 10, "days": 2},
        "CARROT": {"seed": 20, "days": 3},
        "TOMATO": {"seed": 30, "days": 4},
        "STRAWBERRY": {"seed": 50, "days": 5},
        "MELON": {"seed": 90, "days": 6}
    }
    
    for crop, data in crop_specs.items():
        price = market_prices.get(crop, 0.0)
        cost = data["seed"]
        days = data["days"]
        
        # Calculate true daily margins (ROI)
        roi_per_day = (price - cost) / max(1, days)
        if roi_per_day > max_roi_per_day:
            max_roi_per_day = roi_per_day
            best_crop = crop
            
    return best_crop

def smart_agent(obs: Dict[str, Any], config: Optional[Any] = None) -> Dict[str, Any]:
    """
    An enterprise-grade, defensive, macro-optimized agent for Kaggriculture.
    Uses coordinated target assignment matrices and active collision resolution.
    """
    # 1. Guard against incomplete observations or environment failures
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    private = obs.get("private", {}) or {}
    market_layer = obs.get("market", {}) or {}
    market_prices = market_layer.get("prices", {}) or {}
    
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}
        
    farm = farms[player]
    board_size = len(farm.get("tiles", []))
    money = farm.get("money", 0.0)
    
    # 2. Process Storage, Assets, and Inventories Defensively
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    available_seeds = [k for k, v in seeds.items() if v > 0]
    
    market_actions = []
    
    # --- STRATEGY 1: LIQUIDITY STABILIZATION (MARKET ARBITRAGE) ---
    # Intentionally liquidate stocks if market values beat our target threshold
    for item, qty in shed.items():
        if qty > 0:
            price = market_prices.get(item, 0.0)
            if price >= 45.0:  # Competitive floor pricing
                market_actions.append(["SELL", item, int(qty)])
                
    # --- STRATEGY 2: MACRO INVESTMENT & REINVESTMENT ---
    target_crop = _evaluate_crop_roi(market_prices)
    total_owned_seeds = sum(seeds.values()) if seeds else 0
    
    # Maintain seed stocks if liquidity handles it safely
    if total_owned_seeds < 6 and money > 250.0:
        market_actions.append(["BUY_SEED", target_crop, 3])
        money -= 150.0

    # --- STRATEGY 3: SPATIAL ANALYSIS & PRIORITY GRID ---
    harvest_tasks = []
    water_tasks = []
    planting_tiles = []
    
    for y in range(board_size):
        for x in range(board_size):
            tile = farm["tiles"][y][x]
            if tile == "LOCKED":
                continue
                
            if isinstance(tile, dict):
                kind = tile.get("kind", "")
                if kind == "PLANT":
                    # Priority 0: Instant extraction of ripe crop yields
                    if tile.get("yield_units", 0) > 0:
                        harvest_tasks.append((x, y))
                    # Priority 1: Keep plants alive
                    elif not tile.get("watered_today", True):
                        water_tasks.append((x, y))
            elif tile is None:
                planting_tiles.append((x, y))

    # --- STRATEGY 4: MULTI-ENTITY COORDINATED DISPATCH ---
    # Initialize position map tracking to instantly resolve step overlapping
    reserved_positions = set()
    worker_registry = []
    
    # Track the location of every active allied asset
    fx, fy = farm.get("farmer", (0, 0))
    worker_registry.append(("farmer", fx, fy, farm["tiles"][fy][fx]))
    reserved_positions.add((fx, fy))
    
    for hx, hy in farm.get("hands", []):
        worker_registry.append(("hand", hx, hy, farm["tiles"][hy][hx]))
        reserved_positions.add((hx, hy))
        
    farmer_action = ["PASS"]
    hands_actions = []
    
    # Assign actions based on local task proximity
    for role, wx, wy, current_tile in worker_registry:
        action_dispatched = False
        
        # Action Block A: Standing on an actionable item
        if isinstance(current_tile, dict) and current_tile.get("yield_units", 0) > 0:
            final_act = ["HARVEST"]
            action_dispatched = True
        elif isinstance(current_tile, dict) and not current_tile.get("watered_today", True):
            final_act = ["WATER"]
            action_dispatched = True
        elif current_tile is None and available_seeds:
            chosen_seed = max(available_seeds, key=lambda s: market_prices.get(s, 0.0))
            final_act = ["PLANT", chosen_seed]
            action_dispatched = True
            
        # Action Block B: Navigation to the closest spatial objective
        if not action_dispatched:
            # Combine objectives according to structural game priorities
            all_targets = harvest_tasks + water_tasks + (planting_tiles if available_seeds else [])
            
            if all_targets:
                # Dynamic matching: Find the nearest target node on the field map
                best_target = min(all_targets, key=lambda tgt: _manhattan_dist(wx, wy, tgt[0], tgt[1]))
                step_move = _get_path_step(wx, wy, best_target[0], best_target[1], reserved_positions, board_size)
                final_act = [step_move]
            else:
                final_act = ["PASS"]
                
        # Append structured outputs safely into indexing schemas
        if role == "farmer":
            farmer_action = final_act
        else:
            hands_actions.append(final_act)

    # --- STRATEGY 5: SCALE UP PRODUCTION LABOR ---
    total_active_tasks = len(harvest_tasks) + len(water_tasks)
    if total_active_tasks > 6 and money > 600.0 and farm.get("hires_today", 0) < 2:
        market_actions.append(["HIRE_HAND"])

    return {
        "farmer": farmer_action,
        "hands": hands_actions,
        "market": market_actions
    }
