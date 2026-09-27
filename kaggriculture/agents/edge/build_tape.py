"""Build agents/edge/<name>/main.py = tape base + tape_edge_layer with a config dict.
    python agents/edge/build_tape.py NAME TAPE_MAIN 'dict(...)'"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = dict(front=False, fg=False, fg_px=1, fg_until=696, fg_shed=90,
               fg_items=('MILK', 'WOOL', 'STRAWBERRY', 'EGG', 'CARROT', 'TOMATO', 'MELON'),
               liq=False, liq_from=717,
               sa=False, sa_from=216, sa_max=1, sa_min_abs=1, sa_pos='front', reserve_h=48,
               sa_items=('STRAWBERRY', 'MELON', 'MILK', 'WOOL', 'EGG', 'TOMATO', 'CARROT'),
               full_below=None, arm_k=None, arm_before=400, l2=False, l2_model='robust')
def build(name, base, overrides):
    cfg = dict(DEFAULT); cfg.update(overrides)
    src = open(base, encoding='utf-8').read()
    layer = open(os.path.join(HERE, 'tape_edge_layer.py'), encoding='utf-8').read().replace('__TE_CFG__', repr(cfg))
    d = os.path.join(HERE, name); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'main.py'), 'w', encoding='utf-8') as f:
        f.write(src.rstrip('\n') + '\n' + layer)
    return os.path.join(d, 'main.py')
if __name__ == '__main__':
    print(build(sys.argv[1], sys.argv[2], eval(sys.argv[3]) if len(sys.argv) > 3 else {}))
