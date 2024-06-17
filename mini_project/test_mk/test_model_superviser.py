import matplotlib.pyplot as plt
import numpy as np

# Données simplifiées
ages = np.array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
net_worths = np.array([5000, 10000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000])

# Points représentatifs choisis
representative_ages = np.array([30, 60])
representative_net_worths = np.array([40000, 160000])

# Calculer la ligne de régression
line_ages = np.linspace(20, 70, 500)
line_net_worths = 4000 * line_ages - 80000

# Tracer les données et la ligne de régression
plt.scatter(ages, net_worths, color='blue', label='Données')
plt.plot(line_ages, line_net_worths, color='red', label='Ligne de Régression')

# Configuration du graphique
plt.xlabel('Âge')
plt.ylabel('Valeur nette')
plt.title('Relation entre l\'Âge et la Valeur Nette')
plt.legend()
plt.grid(True)
plt.show()
