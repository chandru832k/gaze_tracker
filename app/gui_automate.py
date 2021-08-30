
import time

pag.FAILSAFE = False
pag.PAUSE = 2.5
def do(event_type, coordinate):
	if event_type == "click":
		do('hover', coordinate)
		time.sleep(3)
		pag.click()
	elif event_type == "hover":
		pag.moveTo(*coordinate)

def fill(data, coordinate):
	pag.click(*coordinate)
	pag.write(data)

fill("83389", (804, 364))
print("account number filled")
fill("735", (790, 478))
print("pin filled")
do('click', (829, 546))
print("submit clicked")

do('click', (823, 683))
print("withdraw clicked")
fill("735", (820, 333))
print("amt filled")
do('click', (874, 397))
# print("withdraw success")
# do('click', (847, 743))
# fill("735", (820, 333))
# print("amt filled")
# do('click', (847, 743))
# print("deposit sucess")
# do('click', (858, 845))
