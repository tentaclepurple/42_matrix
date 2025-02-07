from dataclasses import dataclass
from typing import Union


@dataclass 
class Complex:
    real: float
    imag: float

    def __str__(self) -> str:
        return f"({self.real}+{self.imag}i)"

    def __add__(self, other: 'Complex') -> 'Complex':
        return Complex(self.real + other.real, self.imag + other.imag)
    
    def __mul__(self, other: Union['Complex', float]) -> 'Complex':
        if isinstance(other, (int, float)):
            return Complex(self.real * other, self.imag * other)
        return Complex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )
    
    def __sub__(self, other: 'Complex') -> 'Complex':
        return Complex(self.real - other.real, self.imag - other.imag)

    def __truediv__(self, other: 'Complex') -> 'Complex':
        return Complex(
            (self.real * other.real + self.imag * other.imag) / (other.real ** 2 + other.imag ** 2),
            (self.imag * other.real - self.real * other.imag) / (other.real ** 2 + other.imag ** 2)
        )
    
    def __div__(self, other: 'Complex') -> 'Complex':
        return self.__truediv__(other)

    def conjugate(self) -> 'Complex':
        return Complex(self.real, -self.imag)
    
    def __abs__(self) -> float:
        return (self.real * self.real + self.imag * self.imag) ** 0.5


if __name__ == '__main__':
    a = Complex(1, 2)
    b = Complex(3, 4)
    print(a + b)
    print(a - b)
    print(a * b)
    print(a / b)
    print(abs(a))
    print(a.conjugate())
    print()
    print("my Complex")

    print()
    print("Python Complex")
    c = 1 + 2j
    d = 3 + 4j
    print(c + d)
    print(c - d)
    print(c * d)
    print(c / d)
    print(abs(c))
