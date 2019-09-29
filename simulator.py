import DNN
from data import *


NN = DNN.NeuralNetwork(len(battle_data[0]), len(battle_outcomes[0]), rate=0.3)
NN.hidden_layer(8)
NN.hidden_layer(8)
NN.initialize_weights()


def run_simulation(data):
	global training

	troop_ratio = int(data[1])/int(data[0])
	data.pop(1)
	data[0] = troop_ratio

	idx_list = []
	for i, item in enumerate(data):
		if item != '':
			idx_list.append(i)
		if data[0] == '':
			return 'Not Enough Data'

	training = [[] for row in battle_data]
	for idx in idx_list:
		for i, row in enumerate(battle_data):
			training[i].append(battle_data[idx])

	NN.train(battle_data, battle_outcomes, gd_type='mini batch', epochs=50, graph=False)
	NN.save_model()

	if 'howe' in data[1].lower():
		data[1] = 1
	else:
		data[1] = 0

	if 'brit' in data[2].lower():
		data[2] = 0
	else:
		data[2] = 1

	if 'french' in data[3].lower() and 'spanish' in data[3].lower():
		data[3] = 2
	elif 'french' in data[3].lower() or 'spanish' in data[3].lower():
		data[3] = 1
	else:
		data[3] = 0

	NN_data = NN.load_model()
	prediction = NN.predict(data, NN_data)[0][0]

	print(prediction)

	if float(prediction) > float(.5):
		prediction = 'Colonists'
	else:
		prediction = 'British'

	return prediction







