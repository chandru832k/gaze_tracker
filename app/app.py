from flask import Flask, render_template, redirect, request, url_for, flash
from models import Account, db, app


@app.route("/", methods=['POST', 'GET'])
def index():
	if request.method == 'GET':
		err = None
	if request.method == 'POST':
		if not request.form['account-number'] and not request.form['pin']:
			err = "All fields are required!!!"
		else :
			account_number = request.form['account-number']
			account = Account.query.filter_by(account_number=account_number).first()
			if not account or account.pin != int(request.form['pin']):
				err = "Invalid Account number or pin!!!"
			else:
				return redirect(url_for('menu', account_number=account_number))

	return render_template('index.html', error=err)

@app.route("/menu/<int:account_number>/")
def menu(account_number):
    account = Account.query.filter_by(account_number=account_number).first()
    return render_template('menu.html', account=account)

@app.route("/menu/<int:account_number>/go_to_withdrawal")
def redirecting_with(account_number):
	return redirect(url_for('withdraw', account_number=account_number))

@app.route("/menu/<int:account_number>/go_to_deposit")
def redirecting_dep(account_number):
	return redirect(url_for('deposit', account_number=account_number))

@app.route("/withdrawal/<int:account_number>", methods=['POST', 'GET'])
def withdraw(account_number):
	err=None
	if request.method == 'POST':
		if not request.form.get('amount'):
			err = "Invalid amount!!!"
		else:
			amount = int(request.form.get('amount'))
			acc = Account.query.filter_by(account_number=account_number).first()
			if amount > acc.balance :
				err = " Low Balance!!! "
			elif amount > 5000:
				err = "Enter less than 5000"
			elif amount < 1 :
				err= "Invalid amount!!!"
			else :
				acc.balance -= amount
				db.session.add(acc)
				db.session.commit()
				return redirect(url_for('menu', account_number=account_number))
	return render_template('withdrawal.html', error=err)

@app.route("/deposit/<int:account_number>", methods=['POST', 'GET'])
def deposit(account_number):
	err = None
	if request.method == 'POST':
		if not request.form.get('amount'):
			err = 'Invalid deposit amount!!!'
		else:
			amount = int(request.form.get('amount'))
			acc = Account.query.filter_by(account_number=account_number).first()
			if amount > 10000:
				err = "Enter less than 10000"
			elif amount < 1 :
				err= "Invalid deposit amount!!!"
			else :
				acc.balance += amount
				db.session.add(acc)
				db.session.commit()
				return redirect(url_for('menu', account_number=account_number))

	return render_template('deposit.html', error=err)

@app.route('/menu/<int:account_number>/exit')
def exit(account_number):
	return redirect(url_for('index'))
app.run(debug=True)