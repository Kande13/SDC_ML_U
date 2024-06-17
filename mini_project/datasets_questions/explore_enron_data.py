#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib
import os 
import sys
import pickle as pkl

finalProject_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../final_project'))
sys.path.append(finalProject_path)

with open(f"{finalProject_path}/final_project_dataset.pkl", "rb") as fpd:
    enron_data = joblib.load(fpd)

print(enron_data)

# Lecture du fichier en mode texte
with open(f"{finalProject_path}/poi_names.txt", "r") as file:
    poi_names = file.readlines()

# Traitement des noms
#poi_names = [line.strip() for line in poi_names if line.strip()]
poi_count = sum(1 for line in poi_names if line.strip().startswith("(y)"))

# print(f"Nombre total de POI : {poi_count}")
# print(len(poi_names))

nb_pois, nb_Salary, nb_email, nb_salary_email = 0, 0, 0, 0
for person_name, features in enron_data.items():

    if enron_data[person_name]["poi"] == True:
        nb_pois += 1
    if enron_data[person_name]['salary'] != 'NaN':
        nb_Salary += 1
    if enron_data[person_name]['email_address'] != 'NaN':
        nb_email += 1
    if enron_data[person_name]['salary'] != 'NaN' and enron_data[person_name]['email_address'] != 'NaN':
        nb_salary_email +=1


print(f"{person_name} : {len(features)} functionality")
print(f"Number of PIO: {nb_pois}")
print(f"Number of Salary: {nb_Salary}")
print(f"Number of Email: {nb_email}")
print(f"Number of Email: {nb_salary_email}")

count_salary = sum(1 for person in enron_data.values() if person.get('salary') != 'NaN')
count_email = sum(1 for person in enron_data.values() if person.get('email_address') != 'NaN')
print(f"Nombre de personnes avec un salaire quantifié : {count_salary}")
print(f"Nombre de personnes avec une adresse email connue : {count_email}")

if 'COLWELL WESLEY' in enron_data:
    messages_to_poi = enron_data['COLWELL WESLEY']['from_this_person_to_poi']
    print(f"Nombre de messages envoyés par Wesley Colwell aux POI : {messages_to_poi}")

if 'SKILLING JEFFREY K' in enron_data:
    exercised_stock_options = enron_data['SKILLING JEFFREY K']['exercised_stock_options']
    print(f"Valeur des options d'achat d'actions exercées par Jeff Skilling : {exercised_stock_options}")
 
count_nan_total_payments = sum(1 for person in enron_data.values() if person.get('total_payments') == 'NaN')
total_people = len(enron_data)

# Calculate the percentage of people with 'NaN' for total_payments
percentage_nan_total_payments = (count_nan_total_payments / total_people) * 100

print(f"Nombre de NaN et de Pourcentage de payement: {count_nan_total_payments} , {round(percentage_nan_total_payments, 2)}")

# Compter le nombre de POIs avec 'NaN' pour total_payments
poi_nan_total_payments = sum(1 for person in enron_data.values() if person.get('poi') == True and person.get('total_payments') == 'NaN')
total_pois = sum(1 for person in enron_data.values() if person.get('poi') == True)

# Calculer le pourcentage de POIs avec 'NaN' pour total_payments
percentage_poi_nan_total_payments = (poi_nan_total_payments / total_pois) * 100

print(f"Nombre de POI avec 'NaN' pour total_payments : {poi_nan_total_payments}")
print(f"Pourcentage de POI avec 'NaN' pour total_payments : {percentage_poi_nan_total_payments:.2f}%")


# Ajouter 10 nouveaux POI avec 'NaN' pour total_payments
for i in range(10):
    enron_data[f"NEW_POI_{i}"] = {
        "salary": "NaN",
        "total_payments": "NaN",
        "email_address": f"new_poi_{i}@enron.com",
        "poi": True
    }

# Compter le nombre total de personnes
total_people = len(enron_data)

# Compter le nombre de personnes avec 'NaN' pour total_payments
count_nan_total_payments = sum(1 for person in enron_data.values() if person.get('total_payments') == 'NaN')

# Calculer le pourcentage de personnes avec 'NaN' pour total_payments
percentage_nan_total_payments = (count_nan_total_payments / total_people) * 100

print(f"Nombre total de personnes dans l'ensemble de données : {total_people}")
print(f"Nombre de personnes avec 'NaN' pour total_payments : {count_nan_total_payments}")
print(f"Pourcentage de personnes avec 'NaN' pour total_payments : {percentage_nan_total_payments:.2f}%")

# Compter le nombre total de POI
total_pois = sum(1 for person in enron_data.values() if person.get('poi') == True)

# Compter le nombre de POI avec 'NaN' pour total_payments
poi_nan_total_payments = sum(1 for person in enron_data.values() if person.get('poi') == True and person.get('total_payments') == 'NaN')

print(f"Nombre total de POI dans l'ensemble de données : {total_pois}")
print(f"Nombre de POI avec 'NaN' pour total_payments : {poi_nan_total_payments}")