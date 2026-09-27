"""Build agents/edge/<name>/main.py = base source + edge_layer with a config dict.
    python agents/edge/build.py NAME BASE_MAIN 'dict(sa=True, ...)'
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = dict(sa=False, sa_from=216, sa_to=718, sa_mod4=(0, 1, 2, 3), sa_items=('STRAWBERRY', 'MELON', 'MILK', 'WOOL', 'EGG', 'TOMATO', 'CARROT'),
               sa_held_only=False, sa_max=100, sa_min_ratio=0.0, sa_min_abs=1, sa_front=False,
               sells_first=False, l2=False, l2_from=0)
def build(name, base, overrides):
    cfg = dict(DEFAULT); cfg.update(overrides)
    src = open(base, encoding='utf-8').read()
    layer = open(os.path.join(HERE, 'edge_layer.py'), encoding='utf-8').read().replace('__EDGE_CFG__', repr(cfg))
    d = os.path.join(HERE, name); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'main.py'), 'w', encoding='utf-8') as f:
        f.write(src.rstrip('\n') + '\n' + layer)
    return os.path.join(d, 'main.py')
if __name__ == '__main__':
    print(build(sys.argv[1], sys.argv[2], eval(sys.argv[3]) if len(sys.argv) > 3 else {}))
