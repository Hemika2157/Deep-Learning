
# Experiment 24: Graphical Model Sampling and Probabilistic Inference

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling


# ------------------------------------------------
# 1. CREATE THE BAYESIAN NETWORK
# ------------------------------------------------

model = DiscreteBayesianNetwork([
    ('Smoking', 'LungDisease'),
    ('Pollution', 'LungDisease'),
    ('LungDisease', 'Cough'),
    ('LungDisease', 'Xray'),
    ('LungDisease', 'Fatigue')
])

# Prior probabilities
cpd_smoking = TabularCPD(
    variable='Smoking',
    variable_card=2,
    values=[[0.7], [0.3]]
)

cpd_pollution = TabularCPD(
    variable='Pollution',
    variable_card=2,
    values=[[0.8], [0.2]]
)

# Lung Disease probabilities
cpd_disease = TabularCPD(
    variable='LungDisease',
    variable_card=2,
    values=[
        [0.99, 0.90, 0.90, 0.30],
        [0.01, 0.10, 0.10, 0.70]
    ],
    evidence=['Smoking', 'Pollution'],
    evidence_card=[2, 2]
)

# Cough probabilities
cpd_cough = TabularCPD(
    variable='Cough',
    variable_card=2,
    values=[
        [0.8, 0.2],
        [0.2, 0.8]
    ],
    evidence=['LungDisease'],
    evidence_card=[2]
)

# X-ray probabilities
cpd_xray = TabularCPD(
    variable='Xray',
    variable_card=2,
    values=[
        [0.9, 0.1],
        [0.1, 0.9]
    ],
    evidence=['LungDisease'],
    evidence_card=[2]
)

# Fatigue probabilities
cpd_fatigue = TabularCPD(
    variable='Fatigue',
    variable_card=2,
    values=[
        [0.7, 0.3],
        [0.3, 0.7]
    ],
    evidence=['LungDisease'],
    evidence_card=[2]
)

model.add_cpds(
    cpd_smoking,
    cpd_pollution,
    cpd_disease,
    cpd_cough,
    cpd_xray,
    cpd_fatigue
)

print("Bayesian Network Valid:", model.check_model())


# ------------------------------------------------
# 2. VISUALIZE THE NETWORK
# ------------------------------------------------

G = nx.DiGraph()
G.add_edges_from(model.edges())

pos = {
    'Smoking': (0, 2),
    'Pollution': (2, 2),
    'LungDisease': (1, 1),
    'Cough': (0, 0),
    'Xray': (1, 0),
    'Fatigue': (2, 0)
}

plt.figure(figsize=(8, 5))
nx.draw(
    G, pos,
    with_labels=True,
    node_color='lightblue',
    node_size=2500,
    font_size=9,
    arrows=True,
    arrowsize=20
)
plt.title("Medical Bayesian Network")
plt.show()


# ------------------------------------------------
# 3. EXACT INFERENCE USING VARIABLE ELIMINATION
# ------------------------------------------------

inference = VariableElimination(model)

evidence_cases = {
    "Cough=Yes, Xray=Abnormal": {
        'Cough': 1,
        'Xray': 1
    },

    "Cough=Yes, Xray=Abnormal, Smoking=Yes": {
        'Cough': 1,
        'Xray': 1,
        'Smoking': 1
    },

    "Cough=Yes, Xray=Abnormal, Smoking=No": {
        'Cough': 1,
        'Xray': 1,
        'Smoking': 0
    }
}

exact_results = {}

print("\nEXACT INFERENCE RESULTS")

for name, evidence in evidence_cases.items():

    result = inference.query(
        variables=['LungDisease'],
        evidence=evidence,
        show_progress=False
    )

    probability = result.values[1]
    exact_results[name] = probability

    print(name)
    print("Probability of Lung Disease:", round(probability, 4))
    print()


# ------------------------------------------------
# 4. FORWARD SAMPLING
# ------------------------------------------------

sampler = BayesianModelSampling(model)

samples = sampler.forward_sample(
    size=200000,
    seed=42,
    show_progress=False
)

# Convert samples into a Pandas DataFrame
df = pd.DataFrame(samples)

print("FORWARD SAMPLING")
print(df.head())

print("\nGenerated Samples:", len(df))


# ------------------------------------------------
# 5. REJECTION SAMPLING USING PANDAS
# ------------------------------------------------

def rejection_sampling(data, evidence):

    # Keep only rows matching all evidence conditions
    selected = data.copy()

    for variable, value in evidence.items():
        selected = selected[selected[variable] == value]

    if len(selected) == 0:
        return np.nan, 0

    # Estimate P(LungDisease=Yes | evidence)
    probability = (selected['LungDisease'] == 1).mean()

    return probability, len(selected)


print("\nREJECTION SAMPLING RESULTS")

approx_results = {}

for name, evidence in evidence_cases.items():

    probability, count = rejection_sampling(df, evidence)

    approx_results[name] = probability

    print(name)
    print("Matching Samples:", count)
    print("Estimated Probability:", round(probability, 4))
    print()


# ------------------------------------------------
# 6. COMPARE EXACT AND APPROXIMATE RESULTS
# ------------------------------------------------

comparison = pd.DataFrame({
    'Evidence': list(evidence_cases.keys()),
    'Exact Probability': [
        exact_results[name] for name in evidence_cases
    ],
    'Approximate Probability': [
        approx_results[name] for name in evidence_cases
    ]
})

comparison['Absolute Error'] = abs(
    comparison['Exact Probability'] -
    comparison['Approximate Probability']
)

print("\nEXACT VS APPROXIMATE INFERENCE")
print(comparison.to_string(index=False))


# ------------------------------------------------
# 7. CONVERGENCE ANALYSIS
# ------------------------------------------------

sample_sizes = [1000, 5000, 10000, 25000, 50000, 100000, 200000]

# Use the first evidence case for convergence
convergence_evidence = evidence_cases[
    "Cough=Yes, Xray=Abnormal"
]

estimates = []

for size in sample_sizes:

    subset = df.iloc[:size]

    probability, count = rejection_sampling(
        subset,
        convergence_evidence
    )

    estimates.append(probability)

print("\nCONVERGENCE RESULTS")

for size, estimate in zip(sample_sizes, estimates):
    print("Sample Size:", size,
          "Estimated Probability:", round(estimate, 4))


# Plot convergence
plt.figure(figsize=(8, 5))

plt.plot(
    sample_sizes,
    estimates,
    marker='o',
    label='Rejection Sampling'
)

plt.axhline(
    y=exact_results["Cough=Yes, Xray=Abnormal"],
    linestyle='--',
    label='Exact Probability'
)

plt.xlabel("Number of Samples")
plt.ylabel("Probability of Lung Disease")
plt.title("Convergence of Probabilistic Inference")
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------------------------
# 8. BAR CHART COMPARISON
# ------------------------------------------------

x = np.arange(len(comparison))
width = 0.35

plt.figure(figsize=(10, 5))

plt.bar(
    x - width / 2,
    comparison['Exact Probability'],
    width,
    label='Exact Inference'
)

plt.bar(
    x + width / 2,
    comparison['Approximate Probability'],
    width,
    label='Approximate Inference'
)

plt.xticks(
    x,
    ['Without Smoking', 'Smoking = Yes', 'Smoking = No'],
    rotation=10
)

plt.ylabel("Probability of Lung Disease")
plt.title("Exact vs Approximate Probability")
plt.legend()
plt.tight_layout()
plt.show()

print("\nExperiment 24 Completed Successfully!")

