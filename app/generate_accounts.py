from models import Account, db
from random import randint, choices


def name_generator(width=5):
	temp = choices(range(97, 123), k=width)
	return ''.join(map(chr, temp))

def generate_some_accounts(n):
	for _ in range(n):
		temp = Account(name=name_generator(5), account_number=randint(a=10000, b=99999), pin=randint(a=100, b=999), balance=randint(100, 10000))
		db.session.add(temp)
	db.session.commit()

print(*list(Account.query.all()), sep='\n')