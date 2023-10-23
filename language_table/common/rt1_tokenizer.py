import tensorflow_hub as hub


def tokenize_text(text):
  """Tokenizes the input text given a tokenizer."""
  embed = hub.load("/mnt/ve_share2/zy/robotics_transformer/data_add_language_embedding/Universal_Sentence_Encoder")
  tokens = embed([text])
  return tokens

# text ='push the blue triangle closer to yellow heart'
# print(tokenize_text(text))
