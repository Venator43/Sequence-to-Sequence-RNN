import tensorflow as tf
import numpy as np

x = []
y = []
x_character = set()
y_character = set()

dataset_path = "fra.txt"

with open(dataset_path,'r',encoding = "utf-8") as f:
	full_data = f.read().split("\n")

for data in full_data[:50000]:
	if data != "":
		#print(data.split('\t'))
		raw_x, raw_y, _ = data.split("\t")
		raw_y = "\t" + raw_y + "\n"
		
		x.append(raw_x)
		y.append(raw_y)
	
		for char in raw_x:
			if char not in x_character:
				x_character.add(char)
		for char in raw_y:
			if char not in y_character:
				y_character.add(char)

x_character = sorted(list(x_character))
y_character = sorted(list(y_character))
num_encoder_token = len(x_character)
num_decoder_token = len(y_character)
max_x_len = max([len(txt) for txt in x])
max_y_len = max([len(txt) for txt in y])

print("Number Of Training Data : ", len(x))
print("Number Of Input Token : ", len(x_character))
print("Number Of Output Token : ", len(y_character))
print("Max Input Sequence Length : ", max_x_len)

x_char_index = dict([(char,i) for i,char in enumerate(x_character)])
y_char_index = dict([(char,i) for i,char in enumerate(y_character)])

encoder_x = np.zeros((len(x),max_x_len,num_encoder_token),dtype = 'float32')
decoder_x = np.zeros((len(y),max_y_len,num_decoder_token),dtype = 'float32')
decoder_y = np.zeros((len(y),max_y_len,num_decoder_token),dtype = 'float32')

for i, (x_data,y_data) in enumerate(zip(x,y)):
	for t,char in enumerate(x_data):
		encoder_x[i,t,x_char_index[char]] = 1.
		encoder_x[i,t+1:,x_char_index[" "]] = 1.
	for t,char in enumerate(y_data):
		decoder_x[i,t,y_char_index[char]] = 1.
		if t > 0:
			decoder_y[i,t-1,y_char_index[char]] = 1.

	decoder_x[i,t + 1:,y_char_index[" "]] = 1.
	decoder_y[i,t:,y_char_index[" "]] = 1.

print(encoder_x[0].shape)

encoder_inputs = tf.keras.layers.Input(shape=(None,num_encoder_token))
encoder = tf.keras.layers.LSTM(130,return_state=True)
encoder_output,state_h,state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input(shape=(None,num_decoder_token))
decoder_lstm = tf.keras.layers.LSTM(130, return_sequences=True, return_state=True)
decoder_output,_,_ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(num_decoder_token,activation = "softmax")
decoder_output = decoder_dense(decoder_output)

model = tf.keras.models.Model([encoder_inputs,decoder_inputs],decoder_output)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.fit([encoder_x,decoder_x],decoder_y,batch_size=20,epochs=20,validation_split=0.2)

encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

decoder_state_x_h = tf.keras.layers.Input(shape=(130,))
decoder_state_x_c = tf.keras.layers.Input(shape=(130,))
decoder_state_x = [decoder_state_x_h,decoder_state_x_c]
decoder_output, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_x)
decoder_state = [state_h,state_c]
decoder_output = decoder_dense(decoder_output)
decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_state_x, [decoder_output] + decoder_state)

reverse_input_char_index = dict((i,char) for char,i in x_char_index.items())
reverse_target_char_index = dict((i,char) for char,i in y_char_index.items())

def decode_sequence(input_seq):
	# Encode the input as state vectors.
	states_value = encoder_model.predict(input_seq)

	# Generate empty target sequence of length 1.
	target_seq = np.zeros((1, 1, num_decoder_token))
	# Populate the first character of target sequence with the start character.
	target_seq[0, 0, y_char_index['\t']] = 1.

	# Sampling loop for a batch of sequences
	# (to simplify, here we assume a batch of size 1).
	stop_condition = False
	decoded_sentence = ''
	while not stop_condition:
		output_tokens, h, c = decoder_model.predict(
			[target_seq] + states_value)

		# Sample a token
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_target_char_index[sampled_token_index]
		decoded_sentence += sampled_char

		# Exit condition: either hit max length
		# or find stop character.
		if (sampled_char == '\n' or
		   len(decoded_sentence) > max_y_len):
			stop_condition = True

		# Update the target sequence (of length 1).
		target_seq = np.zeros((1, 1, num_decoder_token))
		target_seq[0, 0, sampled_token_index] = 1.

		# Update states
		states_value = [h, c]

	return decoded_sentence

for i in range(100):
	input_seq = encoder_x[i: i + 1]
	decoded_sentence = decode_sequence(input_seq)
	print("Input : ", x[i])
	print("output : ", decoded_sentence)