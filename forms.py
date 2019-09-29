from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField

class InputForm(FlaskForm):
	num_col = StringField('Number of Colonial Troops')
	num_brit = StringField('Number of British Troops')
	general = StringField('British General')
	offensive = StringField('Offensive Side')
	allies = StringField('Colonial Allies')
	button = SubmitField('Run Simulation')