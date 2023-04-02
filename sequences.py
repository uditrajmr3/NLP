# from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

sentences = [
    'Hello World',
    'I love Programming',
    'I love to code',
    'This world is so dramatic',
    'Do I really need to add a new sentence'
]

tokenizer = Tokenizer(num_words = 100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences)
# padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5) // with optional parameters
print(word_index)
print(sequences)
print(word_index)