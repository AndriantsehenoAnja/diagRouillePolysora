# diagRouillePolysora

pip install django

pip install -r ../requirements.txt

pip install opencv-python

python manage.py runserver

# Détection de la rouille Polysora sur les feuilles de maïs


# 1. La brique de base : le Nœud (Node)

Un arbre de décision fonctionne comme un jeu de devinettes (Akinator ou le jeu du portrait).

À chaque étape, il pose une question.

Selon la réponse :

- Gauche : Oui / Inférieur
- Droite : Non / Supérieur

Chaque case de ce jeu est appelée un **Nœud (Node)**.

Il existe deux types de nœuds.

---

## Le nœud de décision

Il pose une question.

Par exemple :

> **Le pourcentage de rouille est-il supérieur à 5 % ?**

Ce nœud mémorise :

- la variable utilisée (`feature_idx`)
- la valeur seuil (`threshold`)
- le nœud de gauche (`left`)
- le nœud de droite (`right`)

---

## Le nœud feuille (Leaf)

Lorsque plus aucune question n'est nécessaire, on arrive dans une feuille.

Elle contient directement la réponse :

- 0 = Feuille saine
- 1 = Rouille Polysora

C'est le verdict final.

---

## L'arbre

Un arbre de décision n'est finalement qu'un ensemble de nœuds reliés entre eux.

```text
          Racine
             │
        Question
        /      \
   Question   Question
    /    \      /    \
   0      1    0      1
```

---

# 2. Le premier robot : l'Arbre de Décision (DecisionTreeMaxMinority)

Ce robot apprend automatiquement à construire toutes les questions en observant les **2 468 images** de feuilles de maïs.

---

## Comment apprend-il ? (`_build_tree`)

Il commence tout en haut de l'arbre avec **toutes les images**.

Puis il applique toujours la même méthode.

### Étape 1 : Vérifier si le groupe est déjà évident

Il regarde les images présentes.

Si elles sont toutes malades :

- il crée immédiatement une feuille "Malade".

Si elles sont toutes saines :

- il crée une feuille "Saine".

On dit alors que le nœud est **pur**.

---

### Étape 2 : Chercher la meilleure question

Si le groupe contient à la fois des feuilles saines et malades, il faut poser une question.

Il teste alors toutes les variables disponibles :

- `pct_rouille`
- `rugosite`
- toute autre variable

Pour chacune, il cherche le meilleur seuil.

Exemple :

```text
pct_rouille ≤ 4.2 ?
```

Le calcul du meilleur seuil est effectué par le script :

```text
meilleursp.py
```

L'objectif est de séparer le plus proprement possible les feuilles malades des feuilles saines.

---

### Étape 3 : Couper le groupe

Une fois la meilleure question trouvée :

```text
pct_rouille ≤ 4.2
```

Les images sont séparées en deux groupes.

Groupe de gauche :

```text
pct_rouille ≤ 4.2
```

Groupe de droite :

```text
pct_rouille > 4.2
```

---

### Étape 4 : Recommencer

Le robot applique exactement la même logique :

- sur le groupe de gauche
- puis sur le groupe de droite

Encore et encore jusqu'à atteindre la profondeur maximale (`max_depth`).

On appelle cela la **récursion**.

---

# Comment l'arbre prédit une nouvelle image ? (`predict`)

Lorsqu'une nouvelle feuille arrive, elle entre par le sommet de l'arbre.

Chaque nœud regarde sa valeur.

Exemple :

```text
pct_rouille = 6 %

Question :

6 ≤ 4.2 ?

Non
```

La feuille part donc à droite.

Puis une autre question est posée.

Et encore une autre.

Jusqu'à atteindre une feuille terminale.

Exemple :

```text
Classe = 1
```

Le verdict est alors :

> **La feuille est atteinte de la rouille Polysora.**

---

# 3. Le super robot : la Forêt Aléatoire (RandomForestMaxMinority)

Un seul arbre possède un défaut.

Il peut devenir **trop spécialisé**.

Par exemple, une seule image possède un reflet de lumière.

L'arbre peut créer une règle uniquement pour cette image.

Il apprend donc par cœur le jeu d'entraînement.

C'est ce qu'on appelle le **surapprentissage (Overfitting)**.

---

## La solution : plusieurs arbres

Au lieu de faire confiance à un seul arbre, on construit une forêt composée de **15 arbres différents**.

Chaque arbre donnera son opinion.

---

# L'astuce magique : le Bagging (`fit`)

Si tous les arbres apprennent avec exactement les mêmes images, ils deviendront identiques.

La forêt ne servirait donc à rien.

Pour éviter cela, on utilise le **Bagging**.

---

## Comment fonctionne le Bagging ?

Imaginons un sac contenant les **2 468 images**.

Pour le premier arbre :

on pioche 2 468 images au hasard.

Mais avec **remplacement**.

Cela signifie que :

- une image peut être tirée plusieurs fois ;
- certaines images ne seront jamais tirées.

Puis on recommence.

- Arbre 2 : nouvel échantillon.
- Arbre 3 : encore un autre.
- ...
- Jusqu'au quinzième arbre.

Chaque arbre reçoit donc un jeu de données légèrement différent.

Ils développent ainsi des comportements différents.

Par exemple :

- Arbre 1 devient expert de `pct_rouille`.
- Arbre 2 préfère `rugosite`.
- Arbre 3 découvre d'autres règles.

Cette diversité rend la forêt beaucoup plus robuste.

---

# Le vote majoritaire (`predict`)

Quand une nouvelle photo est envoyée depuis Django, elle est transmise simultanément aux **15 arbres**.

Chaque arbre vote.

Exemple :

```text
[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

Ici :

- 14 arbres disent : **Malade**
- 1 arbre dit : **Saine**

Le programme calcule alors :

```python
np.mean(votes)
```

La moyenne est très proche de :

```text
1
```

Puis il applique :

```python
np.round(...)
```

Le résultat final devient :

```text
1
```

La forêt entière conclut donc :

> **La feuille est atteinte de la rouille Polysora.**

---

# Pourquoi une forêt est-elle meilleure qu'un seul arbre ?

- Plusieurs avis sont plus fiables qu'un seul.
- Les erreurs individuelles sont compensées.
- Le risque de surapprentissage diminue fortement.
- Les prédictions sont beaucoup plus robustes.

---

# Résultat final

Grâce à cette combinaison de :

- 15 arbres de décision ;
- Bagging ;
- Vote majoritaire.

```
python3 -m config.formulaire.tests

``