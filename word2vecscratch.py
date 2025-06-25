text = '''Machine learning is the study of computer algorithms that \
improve automatically through experience. It is seen as a \
subset of artificial intelligence. Machine learning algorithms \
build a mathematical model based on sample data, known as \
training data, in order to make predictions or decisions without \
being explicitly programmed to do so. Machine learning algorithms \
are used in a wide variety of applications, such as email filtering \
and computer vision, where it is difficult or infeasible to develop \
conventional algorithms to perform the needed tasks.'''

import re
import numpy as np
from vector_store import VectorStore


def tokenize(text):#Creates tokens Using
    pattern = re.compile(
        r"\b[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*\b"  # Words, contractions, hyphens
        r"|[#@][\w-]+"                             # Hashtags and mentions
        r"|(?:[A-Z]\.){2,}"                        # Acronyms like U.S.A.
        r"|\$?\d+(?:\.\d+)?%?"                     # Numbers, currency, percent
    )
    return pattern.findall(text.lower())

tokens = tokenize(text)

def mapping(tokens):#Assigning a link/map between tokens and indices and vice versa
    word_to_id = {}
    id_to_word = {}

    for i, token in enumerate(sorted(set(tokens))):
        word_to_id[token] = i
        id_to_word[i] = token
    return word_to_id, id_to_word

word_to_id, id_to_word = mapping(tokens)
# Step 3: Convert tokens to ids
token_ids = [word_to_id[token] for token in tokens]

# Step 4: Create an embedding matrix
vocab_size = len(word_to_id)
embedding_dim = 5  # you can choose any dimension
embedding_matrix = np.random.randn(vocab_size, embedding_dim)

# Step 5: Perform embedding lookup
embedded_tokens = [embedding_matrix[token_id] for token_id in token_ids]

# Optional: convert to NumPy array
embedded_tokens = np.array(embedded_tokens)

print("Embedding matrix shape:", embedding_matrix.shape)
print("Embedded token shape:", embedded_tokens.shape)
print("Embedded tokens:\n", embedded_tokens)



np.random.seed(42)

def generate_training_data(tokens,word_to_id,window):#Generate training data
    X=[]
    y=[]
    n_tokens = len(tokens)

    for i in range(n_tokens):

        idx = concat(
            range(max(0,i-window),i),
            range(i,min(n_tokens, i + window + 1))
            )
        for j in idx:
            if i == j:
                continue
            X.append(word_to_id[tokens[i]])
            y.append(word_to_id[tokens[j]])

    return np.array(X), np.array(y)

def concat(*iterables):#Used to combine 2 range() objects
    for iterable in iterables:
        yield from iterable

X, y = generate_training_data(tokens, word_to_id, 2)

#print(X.shape)
#print(y.shape)


def init_network(vocab_size, n_embedding):#Model(Represented by Dictionary)and weights
    model = {
        "w1": np.random.randn(vocab_size, n_embedding),
        "w2": np.random.randn(n_embedding, vocab_size)
    }
    return model

model = init_network(len(word_to_id), 10)#model with dimensionality of 10


def forward(model, X_inndices, return_cache=True):#Forward Propagation
    cache = {}
    
    cache["a1"] = model["w1"][X_indices]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])
    
    return cache if return_cache else cache["z"]

def softmax(X):
    res = []
    for x in X:
        exp = np.exp(x)
        res.append(exp / exp.sum())
    return res

#print((X @ model["w1"]).shape)
#print((X @ model["w1"] @ model["w2"]).shape)


def backward(model, X_indices, y_indices, alpha):#Backpropagation
    cache  = forward(model, X)

    batch_size = len(X_indices)
    vocab_size = model["w2"].shape[1]

    y_onehot = np.zeros((batch_size, vocab_size))
    y_onehot[np.arange(batch_size), y_indices] = 1
    
    da2 = cache["z"] - y_onehot
    dw2 = cache["a1"].T @ da2
    
    da1 = da2 @ model["w2"].T
    dw1 = np.zeros_like(model["w1"])
    for i, idx in enumerate(X_indices):
        dw1[idx] += da1[i]

    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    
    return cross_entropy(cache["z"], y_onehot)

def cross_entropy(z, y):#Cross Entropy Loss
    return - np.sum(np.log(z) * y)


#Training and testing Model
import matplotlib.pyplot as plt
import seaborn as sns  # Seaborn adds nice themes and plot functions

sns.set_theme(style="darkgrid")  # Apply a Seaborn style (optional)

n_iter = 50
learning_rate = 0.05

X_indices, y_indices = generate_training_data(tokens, word_to_id, window=2)
model = init_network(len(word_to_id), 10)
history = [backward(model, X_indices, y_indices, learning_rate) for _ in range(n_iter)]


# Plot using Seaborn's default theme and Matplotlib's interface
plt.figure(figsize=(8, 5))
sns.lineplot(x=range(len(history)), y=history, color="skyblue")

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Over Iterations")
plt.tight_layout()
plt.show()

vector_store = VectorStore()
for word, idx in word_to_id.items():
    embedding_vector = model["w1"][idx]
    vector_store.add_vector(word, embedding_vector)

query_word = "machine"
query_vector = model["w1"][word_to_id[query_word]]

similar_words = vector_store.find_similar_vectors(query_vector, num_results=5)
print(f"Top similar words to '{query_word}':")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")
    
#help from https://jaketae.github.io/study/word2vec/
