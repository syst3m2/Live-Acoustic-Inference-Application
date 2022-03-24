from collections import namedtuple
from math import floor

time_tuple = namedtuple('time_tuple', 'sec usec')

class IntegerTimestamp():
    def __init__(self, sec = 0, usec = 0):
        if type(sec)==float:
            self.time = time_tuple(*self.float2intstamp(sec)) #todo: as is, if timestamp is supplied as float, usec is ignored
        else:
            self.time = time_tuple(sec, usec)
        self.time = self.usec_rollover().time

    def __eq__(self, other):
        other = self.convert(other)
        return self.time == other.time

    def __ne__(self, other):
        other = self.convert(other)
        return self.time != other.time

    def __lt__(self, other):
        other=self.convert(other)
        return float(self) < float(other)

    def __le__(self, other):
        other = self.convert(other)
        return float(self) <= float(other)

    def __gt__(self, other):
        other=self.convert(other)
        return float(self) > float(other)

    def __ge__(self, other):
        other = self.convert(other)
        return float(self) >= float(other)

    def __add__(self, other):
        other = self.convert(other)
        return IntegerTimestamp(self.time.sec + other.time.sec, self.time.usec + other.time.usec)

    def __sub__(self, other):
        other = self.convert(other)
        return IntegerTimestamp(self.time.sec - other.time.sec, self.time.usec - other.time.usec)

    def __mul__(self, other):
        if type(other) not in [int, float]:
            raise TypeError('Only multiplication by integers and floats supported at this time')
        return IntegerTimestamp(self.time.sec*other, self.time.usec*other)

    def __truediv__(self, other):
        if type(other) not in [int, float]:
            raise TypeError('Only division by integers and floats supported at this time')
        return IntegerTimestamp(self.time.sec/other, self.time.usec/other)

    def __abs__(self):
        if self<0:
            return IntegerTimestamp(abs(self.time.sec), abs(1000000 - self.time.usec))
        return self
    def __float__(self):
        return self.time.sec + self.time.usec/1000000.0

    def __int__(self):
        return self.time.sec

    def __repr__(self):
        return "IntegerTimestamp(sec = {}, usec = {})".format(*self.time)

    def __iter__(self):
        return iter(self.time)

    def convert(self, other):
        if type(other) == float:
            other = IntegerTimestamp(*self.float2intstamp(other))
        if type(other)== int:
            other = IntegerTimestamp(sec=other, usec=0)
        if type(other) == tuple:
            other = IntegerTimestamp(*other)
        if type(other) == dict:
            other = IntegerTimestamp(**other)
        if type(other)!=IntegerTimestamp:
            raise TypeError('Only addition with floats and other IntegerTimestamps supported at this time')
        return other

    def usec_rollover(self):
        if self.time.usec >= 1000000:
            quotient = self.time.usec/1000000
            self.time = time_tuple(self.time.sec, usec=0)
            self+=quotient
        if self.time.usec < 0:
            quotient = self.time.usec / 1000000
            self.time = time_tuple(floor(self.time.sec+quotient), int((1+ quotient - int(quotient))*1000000))
        return self

    def float2intstamp(self, time_stamp):
        sec = int(time_stamp)
        # usec = int((time_stamp - sec) * 1000000)
        usec =  int('{:.6f}'.format(time_stamp).split('.')[1]) #todo: I *HATE* this hack, but the above produces rounding errors
        return (sec, usec)

    def toJSON(self):
        return dict(self.time._asdict())

def example():
    t1 = IntegerTimestamp(11,999999)
    print(t1)
    print(t1 + IntegerTimestamp(0,3))


if __name__ == '__main__':
    example()