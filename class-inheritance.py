# coding: utf-8

class Unit(object):
    def __init__(self, rank, size, life):
        self.name = self.__class__.__name__
        self.rank = rank
        self.size = size
        self.life = life
        
    def show_status(self):
        print('이름: {}'.format(self.name))
        print('등급: {}'.format(self.rank))
        print('사이즈: {}'.format(self.size))
        print('라이프: {}'.format(self.life))
    def f(self):
        print('super')

class Goblin(Unit):
    def __init__(self, rank, size, life, attack_type):
        super(Goblin, self).__init__(rank, size, life)  # super init
        self.attack_type = attack_type
        
    def show_status(self):
        super(Goblin, self).show_status()
        print('공격 타입: {}'.format(self.attack_type))
    def __call__(self,x):
        print("="*x)
        
goblin_1 = Goblin('병사', 'Small', 100, '근접 공격')

goblin_1.show_status()

goblin_1.f()

goblin_1(10)














