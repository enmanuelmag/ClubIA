import re

re_url = re.compile(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
re_email = re.compile(r'[\w\.-]+@[\w\.-]+')
re_hashtag = re.compile(r'#([^\s]+)')

def clean_str(string):
  string = re_url.sub('', string)
  string = re.sub(r"[^A-Za-z0-9(),!?\'\#@`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  string = string.strip().lower().split()
  string = [ word for word in string if word not in stop_words ]
  return " ".join(string)
