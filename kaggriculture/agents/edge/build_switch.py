"""Build a two-base "switcher" agent: agents/edge/<NAME>/main.py.

    python agents/edge/build_switch.py NAME [--a A_MAIN] [--b B_MAIN] [--default a|b] [--decide 92]

A = cha22-lineage agent (default submissions/p05_d_sb_s150a0/main.py, P05)
B = tetsutani-lineage agent (default submissions/cand_t_sb_s150/main.py, P06)

Each complete agent file (base + edge layer) is embedded zlib/base64-compressed and exec'd into its own
namespace dict, so module globals never collide. Both are fed every observation (identical actions on
the ladder until step 91) until the opponent is classified from the observation at step `decide`:
the rival's public money rose during turn decide-1 (= step 91: the cha22 family sells 3 wheat there, the
tetsutani lineage does not) -> A, otherwise -> B. After the decision only the chosen agent runs.
`switch_agent` is the last callable in the file (Kaggle loads the last callable).
"""
import argparse
import base64
import os
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

TEMPLATE = r'''# Kaggriculture two-base switcher (built by agents/edge/build_switch.py).
# A = {A_NAME} (cha22 lineage + edge), B = {B_NAME} (tetsutani lineage + edge).
# Both embedded agent files are Apache-2.0 derivations; their full notices are retained inside the
# compressed sources below. Both agents see every observation until the opponent is classified at
# step {DECIDE} (rival money rose during turn {DECIDE_M1} -> A, else B); then only the chosen one runs.
import base64 as _sw_b64
import copy as _sw_copy
import zlib as _sw_zlib

_SW_SRC = {{'A': {A_SRC!r}, 'B': {B_SRC!r}}}
_SW_FILE = {{'A': {A_NAME!r}, 'B': {B_NAME!r}}}
_SW_CFG = dict(default={DEFAULT!r}, decide={DECIDE!r}, last_market_from={LMF!r})


def _sw_load(key):
    src = _sw_zlib.decompress(_sw_b64.b64decode(_SW_SRC[key])).decode('utf-8')
    ns = {{'__name__': '__sw_agent_' + key + '__', '__file__': _SW_FILE[key], '__builtins__': __builtins__}}
    exec(compile(src, _SW_FILE[key], 'exec'), ns)
    return [v for v in ns.values() if callable(v)][-1]


_SW_AGENTS = {{k: _sw_load(k) for k in ('A', 'B')}}
_SW_STATE = {{}}
_SW_REPORT = dict(choice=None, decided_at=None, opp_delta=None, diverged_before=None, market_swapped=None, errors=0)
del _SW_SRC


def _sw_opp_money(obs):
    seat = int(obs['player'])
    return float(obs['farms'][1 - seat]['money'])


def switch_agent(observation, configuration=None):
    step = int(observation['step'])
    seat = int(observation['player'])
    st = _SW_STATE.get(seat)
    if st is None or step <= st['step']:
        st = _SW_STATE[seat] = {{'step': -1, 'choice': None, 'm_prev': None, 'clean': True}}
        _SW_REPORT.update(choice=None, decided_at=None, opp_delta=None, diverged_before=None, market_swapped=None, errors=0)
    st['step'] = step
    decide = _SW_CFG['decide']
    if st['choice'] is None and step >= decide:
        choice = _SW_CFG['default']
        try:
            if st['m_prev'] is not None and st['m_prev'][0] == decide - 1:
                delta = _sw_opp_money(observation) - st['m_prev'][1]
                _SW_REPORT['opp_delta'] = delta
                choice = 'A' if delta > 0 else 'B'
        except Exception:
            _SW_REPORT['errors'] += 1
        st['choice'] = choice
        _SW_REPORT.update(choice=choice, decided_at=step)
    try:
        st['m_prev'] = (step, _sw_opp_money(observation))
    except Exception:
        st['m_prev'] = None
    if st['choice'] is not None:
        return _SW_AGENTS[st['choice']](observation, configuration)
    main = _SW_CFG['default']
    other = 'B' if main == 'A' else 'A'
    shadow_obs = _sw_copy.deepcopy(observation)
    act = _SW_AGENTS[main](observation, configuration)
    try:
        alt = _SW_AGENTS[other](shadow_obs, configuration)
        clean = st.get('clean', True)
        if alt != act:
            if _SW_REPORT['diverged_before'] is None:
                _SW_REPORT['diverged_before'] = step
            st['clean'] = False
        # Last pre-decision turn: take the other agent's market list (e.g. skip A's turn-91 wheat sale)
        # when both agents agreed on everything so far and agree on this turn's unit actions.
        if (step == decide - 1 and _SW_CFG['last_market_from'] == other and clean and isinstance(act, dict)
                and isinstance(alt, dict) and act.get('farmer') == alt.get('farmer') and act.get('hands') == alt.get('hands')):
            act = dict(act, market=alt.get('market', []))
            _SW_REPORT['market_swapped'] = step
    except Exception:
        _SW_REPORT['errors'] += 1
    return act


switch_agent.telemetry = _SW_REPORT
'''


def build(name, a, b, default, decide, lmf=None):
    enc = lambda p: base64.b64encode(zlib.compress(open(p, 'rb').read(), 9)).decode('ascii')
    src = TEMPLATE.format(A_NAME=os.path.relpath(a, ROOT), B_NAME=os.path.relpath(b, ROOT), A_SRC=enc(a), B_SRC=enc(b),
                          DEFAULT=default, DECIDE=decide, DECIDE_M1=decide - 1, LMF=lmf)
    d = os.path.join(HERE, name)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, 'main.py')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(src)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--a', default=os.path.join(ROOT, 'submissions/p05_d_sb_s150a0/main.py'))
    ap.add_argument('--b', default=os.path.join(ROOT, 'submissions/cand_t_sb_s150/main.py'))
    ap.add_argument('--default', default='B', choices=['A', 'B'])
    ap.add_argument('--decide', type=int, default=92)
    ap.add_argument('--last-market-from', default=None, choices=['A', 'B'])
    x = ap.parse_args()
    print(build(x.name, os.path.abspath(x.a), os.path.abspath(x.b), x.default, x.decide, x.last_market_from))
