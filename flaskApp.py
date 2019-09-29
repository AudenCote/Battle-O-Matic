from flask import Flask, render_template, url_for, flash, redirect
from forms import InputForm
import simulator

app = Flask(__name__)

app.config['SECRET_KEY'] = '12345'

@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    form = InputForm()
    if form.validate_on_submit():
    	prediction = simulator.run_simulation([form.num_col.data, form.num_brit.data, form.general.data, form.offensive.data, form.allies.data])
    	print(prediction)
    	flash(f'It looks like the {prediction} win this one!', 'info')
    	# return redirect(url_for('home'))

    return render_template('index.html', form = form)


if __name__ == '__main__':
    app.run(debug=True)

