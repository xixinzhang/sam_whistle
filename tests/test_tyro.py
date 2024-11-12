from dataclasses import dataclass
import tyro

@dataclass
class A:
    a: int = 1
    b: str = 'test'

@dataclass
class B:
    c: A = A()

if __name__ == '__main__':
    args = tyro.cli(B)
    print(args)