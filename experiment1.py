
import numpy as np
import matplotlib.pyplot as plt

# 1. Customer-Genre Matrix (6 x 5)
customers = np.array([
    [5, 4, 3, 2, 1],
    [4, 5, 3, 1, 2],
    [1, 2, 5, 4, 3],
    [2, 1, 4, 5, 3],
    [5, 3, 2, 4, 1],
    [1, 3, 4, 2, 5]
])

# 2. Genre-Book Matrix (5 x 4)
books = np.array([
    [0.9, 0.2, 0.1, 0.3],
    [0.8, 0.3, 0.2, 0.1],
    [0.1, 0.9, 0.3, 0.2],
    [0.2, 0.4, 0.9, 0.3],
    [0.3, 0.2, 0.4, 0.9]
])

print("CUSTOMER-GENRE MATRIX:")
print(customers)

print("\nGENRE-BOOK MATRIX:")
print(books)

# 3. Dot Product
dot = np.dot(customers[0], customers[1])
print("\nDot Product of Customer 1 and Customer 2:")
print(dot)

# 4. Matrix Multiplication
scores = np.matmul(customers, books)
print("\nCUSTOMER-BOOK SCORE MATRIX:")
print(scores)

# 5. Transpose
print("\nTranspose of Customer Matrix:")
print(customers.T)

# 6. Inverse and Pseudoinverse
square_matrix = np.array([
    [2, 1],
    [1, 2]
])

print("\nInverse of Square Matrix:")
print(np.linalg.inv(square_matrix))

print("\nPseudoinverse of Customer Matrix:")
print(np.linalg.pinv(customers))

# 7. Element-wise Operations
print("\nElement-wise Addition:")
print(customers + 2)

print("\nElement-wise Multiplication:")
print(customers * customers)

# 8. Cosine Similarity
norms = np.linalg.norm(customers, axis=1, keepdims=True)

normalized = customers / norms
similarity = np.matmul(normalized, normalized.T)

print("\nCOSINE SIMILARITY MATRIX:")
print(np.round(similarity, 3))

# Find two most similar different customers
similarity_without_diagonal = similarity.copy()
np.fill_diagonal(similarity_without_diagonal, -1)

i, j = np.unravel_index(
    np.argmax(similarity_without_diagonal),
    similarity_without_diagonal.shape
)

print("\nMost Similar Customers:")
print("Customer", i + 1, "and Customer", j + 1)
print("Similarity Score:", round(similarity[i, j], 3))

# 9. Dense Layer: y = Wx + b
x = customers[0].reshape(5, 1)
W = books.T
b = np.array([[1], [1], [1], [1]])

y = np.matmul(W, x) + b

print("\nDENSE LAYER OUTPUT:")
print(y)

# 10. Heatmap Visualization
plt.figure(figsize=(8, 5))

plt.imshow(scores, cmap="YlOrRd", aspect="auto")

plt.colorbar(label="Recommendation Score")

plt.xticks(
    range(4),
    ["Book 1", "Book 2", "Book 3", "Book 4"]
)

plt.yticks(
    range(6),
    ["Customer 1", "Customer 2", "Customer 3",
     "Customer 4", "Customer 5", "Customer 6"]
)

plt.xlabel("Books")
plt.ylabel("Customers")
plt.title("Customer-Book Recommendation Score Heatmap")

for row in range(6):
    for col in range(4):
        plt.text(
            col, row,
            f"{scores[row, col]:.2f}",
            ha="center", va="center"
        )

plt.tight_layout()
plt.show()
