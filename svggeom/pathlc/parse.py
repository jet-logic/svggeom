from typing import Optional
from .command import BaseLC, MoveLC


def parse_each(d: str, prev: BaseLC | None = None):
    from re import compile, Match

    d_re = compile(
        r"[\s,]*(?:([MmZzLlHhVvCcSsQqTtAa])|([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?))"
    )
    m: Optional[Match[str]] = None

    def peek():
        # print("peek", m, m and (m.lastindex, m.endpos))
        if m is None:
            n = d_re.search(d)
        else:
            n = d_re.search(d, pos=m.end())
        return n

    def get():
        nonlocal m
        # print("get>", m, m and (m.lastindex, m.endpos))
        if m is None:
            m = d_re.search(d)
        else:
            m = d_re.search(d, pos=m.end())
        # print("<get", m, m and (m.lastindex, m.endpos))
        return m

    def cmd() -> Optional[str]:
        m = get()
        # print("cmd", m, m and (m.lastindex, m.endpos))
        if m:
            v = m.group(1)
            if v:
                return v
            raise ValueError(f"Command expected '{v}' '{d}'")
        return None

    def num():
        v = get().group(2)
        if v:
            return float(v)
        # print("num", m, m and (m.lastindex, m.endpos), m.group(), v)
        raise ValueError(f"Number expected '{v}' '{d}'")

    def is_num():
        x = peek()
        # print("is_num", m, m and (m.lastindex, m.endpos), x and x.group(2))
        return x and x.group(2)

    def vec():
        # print("vec", m, m and (m.lastindex, m.endpos))
        return (num(), num())

    command = None
    first = MoveLC(None, (0, 0))
    cur: BaseLC = prev or first

    while command := cmd():
        if command == "M":
            if cur is first:
                cur = MoveLC(None, vec())
            else:
                cur = cur.M(vec())
            while is_num():
                cur = cur.L(vec())

        elif command == "m":
            if cur is first:
                cur = MoveLC(None, vec())
            else:
                cur = cur.m(vec())
            while is_num():
                cur = cur.l(vec())
        elif command == "Z":
            cur = cur.Z()
        elif command == "z":
            cur = cur.z()
        elif command == "L":
            cur = cur.L(vec())
            while is_num():
                cur = cur.L(vec())
        elif command == "l":
            cur = cur.l(vec())
            while is_num():
                cur = cur.l(vec())
        elif command == "T":
            cur = cur.T(vec())
            while is_num():
                cur = cur.T(vec())
        elif command == "t":
            cur = cur.t(vec())
            while is_num():
                cur = cur.t(vec())
        elif command == "H":
            cur = cur.H(num())
            while is_num():
                cur = cur.H(num())
        elif command == "h":
            cur = cur.h(num())
            while is_num():
                cur = cur.h(num())
        elif command == "V":
            cur = cur.V(num())
            while is_num():
                cur = cur.V(num())
        elif command == "v":
            cur = cur.v(num())
            while is_num():
                cur = cur.v(num())
        elif command == "Q":
            cur = cur.Q(vec(), vec())
            while is_num():
                cur = cur.Q(vec(), vec())
        elif command == "q":
            cur = cur.q(vec(), vec())
            while is_num():
                cur = cur.q(vec(), vec())
        elif command == "S":
            cur = cur.S(vec(), vec())
            while is_num():
                cur = cur.S(vec(), vec())
        elif command == "s":
            cur = cur.s(vec(), vec())
            while is_num():
                cur = cur.s(vec(), vec())
        elif command == "C":
            cur = cur.C(vec(), vec(), vec())
            while is_num():
                cur = cur.C(vec(), vec(), vec())
        elif command == "c":
            cur = cur.c(vec(), vec(), vec())
            while is_num():
                cur = cur.c(vec(), vec(), vec())
        elif command == "A":
            cur = cur.A(num(), num(), num(), num(), num(), vec())
            while is_num():
                cur = cur.A(num(), num(), num(), num(), num(), vec())
        elif command == "a":
            cur = cur.a(num(), num(), num(), num(), num(), vec())
            while is_num():
                cur = cur.a(num(), num(), num(), num(), num(), vec())
        else:
            raise ValueError(f"Invalid path command {command} from '{d}'")

    return cur
