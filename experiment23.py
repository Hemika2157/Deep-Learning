
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Define Bayesian Network structure
print("Creating Bayesian Network...")

model = DiscreteBayesianNetwork([
    ("Smoking", "LungDisease"),
    ("Pollution", "LungDisease"),
    ("LungDisease", "Cough"),
    ("LungDisease", "Xray"),
    ("LungDisease", "Fatigue")
])

# 2. Define Conditional Probability Distributions (CPDs)

# Smoking: 0 = No, 1 = Yes
cpd_smoking = TabularCPD(
    variable="Smoking",
    variable_card=2,
    values=[
        [0.7],
        [0.3]
    ],
    state_names={"Smoking": ["No", "Yes"]}
)

# Pollution: 0 = Low, 1 = High
cpd_pollution = TabularCPD(
    variable="Pollution",
    variable_card=2,
    values=[
        [0.8],
        [0.2]
    ],
    state_names={"Pollution": ["Low", "High"]}
)

# Lung Disease depends on Smoking and Pollution
cpd_lung = TabularCPD(
    variable="LungDisease",
    variable_card=2,
    values=[
        [0.99, 0.90, 0.90, 0.30],
        [0.01, 0.10, 0.10, 0.70]
    ],
    evidence=["Smoking", "Pollution"],
    evidence_card=[2, 2],
    state_names={
        "LungDisease": ["No", "Yes"],
        "Smoking": ["No", "Yes"],
        "Pollution": ["Low", "High"]
    }
)

# Cough depends on Lung Disease
cpd_cough = TabularCPD(
    variable="Cough",
    variable_card=2,
    values=[
        [0.8, 0.2],
        [0.2, 0.8]
    ],
    evidence=["LungDisease"],
    evidence_card=[2],
    state_names={
        "Cough": ["No", "Yes"],
        "LungDisease": ["No", "Yes"]
    }
)

# X-ray result depends on Lung Disease
cpd_xray = TabularCPD(
    variable="Xray",
    variable_card=2,
    values=[
        [0.9, 0.1],
        [0.1, 0.9]
    ],
    evidence=["LungDisease"],
    evidence_card=[2],
    state_names={
        "Xray": ["Normal", "Abnormal"],
        "LungDisease": ["No", "Yes"]
    }
)

# Fatigue depends on Lung Disease
cpd_fatigue = TabularCPD(
    variable="Fatigue",
    variable_card=2,
    values=[
        [0.7, 0.3],
        [0.3, 0.7]
    ],
    evidence=["LungDisease"],
    evidence_card=[2],
    state_names={
        "Fatigue": ["No", "Yes"],
        "LungDisease": ["No", "Yes"]
    }
)

# 3. Add CPDs to the model
model.add_cpds(
    cpd_smoking,
    cpd_pollution,
    cpd_lung,
    cpd_cough,
    cpd_xray,
    cpd_fatigue
)

# 4. Validate Bayesian Network
print("\nChecking Bayesian Network...")

if model.check_model():
    print("Bayesian Network is valid.")

# Display CPDs
print("\nCONDITIONAL PROBABILITY DISTRIBUTIONS")

for cpd in model.get_cpds():
    print("\n", cpd)

# 5. Draw Bayesian Network using NetworkX
G = nx.DiGraph()

G.add_edges_from(model.edges())

positions = {
    "Smoking": (0, 2),
    "Pollution": (2, 2),
    "LungDisease": (1, 1),
    "Cough": (0, 0),
    "Xray": (1, 0),
    "Fatigue": (2, 0)
}

plt.figure(figsize=(10, 7))

nx.draw(
    G,
    positions,
    with_labels=True,
    node_size=3500,
    node_color="lightblue",
    font_size=9,
    font_weight="bold",
    arrows=True,
    arrowsize=20,
    edge_color="black"
)

plt.title("Medical Diagnosis Bayesian Network")
plt.axis("off")
plt.tight_layout()
plt.show()

# 6. Conditional Independence
print("\nCONDITIONAL INDEPENDENCIES")

print(
    "Smoking and Pollution are independent "
    "because they have no connecting parent."
)

print(
    "Cough, Xray and Fatigue are conditionally "
    "independent given LungDisease."
)

print(
    "Smoking and Pollution influence symptoms "
    "through LungDisease."
)

# 7. Find active trails
print("\nACTIVE TRAILS")

trail_queries = [
    ("Smoking", "Cough", []),
    ("Smoking", "Pollution", []),
    ("Cough", "Xray", []),
    ("Smoking", "Cough", ["LungDisease"]),
    ("Cough", "Xray", ["LungDisease"])
]

for start, end, observed in trail_queries:

    trail = model.active_trail_nodes(
        start,
        observed=observed
    )

    if end in trail[start]:
        status = "ACTIVE"
    else:
        status = "BLOCKED"

    print(
        start, "to", end,
        "given", observed,
        ":", status
    )

# 8. Perform probabilistic inference
print("\nPROBABILISTIC INFERENCE")

inference = VariableElimination(model)

# Probability of lung disease given cough
result = inference.query(
    variables=["LungDisease"],
    evidence={"Cough": "Yes"}
)

print("\nLung Disease Probability Given Cough:")
print(result)

# Probability of lung disease given cough and abnormal X-ray
result2 = inference.query(
    variables=["LungDisease"],
    evidence={
        "Cough": "Yes",
        "Xray": "Abnormal"
    }
)

print("\nLung Disease Probability Given Cough and Abnormal X-ray:")
print(result2)

print("\nExperiment 23 completed successfully.")

