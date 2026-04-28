# ============================================================
# src/preprocessing.py
# ============================================================
# Responsabilité : nettoyer le DataFrame maître, créer les
# features, encoder les variables catégorielles, et préparer
# les ensembles Train / Test pour la modélisation.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# ÉTAPE 1 — NETTOYAGE (Cleaning)
# ============================================================

def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit toutes les colonnes de dates (stockées en string)
    au format datetime de pandas.

    Args:
        df: DataFrame maître brut.

    Returns:
        DataFrame avec les dates correctement typées.
    """
    date_columns = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',   # date d'expédition chez le transporteur
        'order_delivered_customer_date',  # date de livraison réelle au client
        'order_estimated_delivery_date',  # date de livraison promise
    ]

    for col in date_columns:
        if col in df.columns:
            # errors='coerce' → transforme les valeurs non parsables en NaT (Not a Time)
            df[col] = pd.to_datetime(df[col], errors='coerce')

    print(f"  ✅ {len(date_columns)} colonnes converties en datetime")
    return df


def filter_valid_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
        Conserve uniquement les commandes avec un statut pertinent pour
        le problème de Return/Refund propensity.

        On conserve les commandes :
            - delivered
            - canceled
            - unavailable

        Les statuts techniques ou incomplets sont retirés.

    Args:
        df: DataFrame avec dates converties.

    Returns:
        DataFrame filtré.
    """
    valid_statuses = ['delivered', 'canceled', 'unavailable']
    initial_count = len(df)
    df = df[df['order_status'].isin(valid_statuses)].copy()
    removed = initial_count - len(df)
    print(f"  ✅ {removed:,} commandes hors périmètre supprimées → {len(df):,} commandes restantes")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
        Gère les valeurs manquantes (NaN) de chaque colonne selon
        la stratégie la plus adaptée :
      - Numérique : imputation par la médiane (robuste aux outliers)
      - Catégoriel : imputation par 'unknown'

    Args:
        df: DataFrame filtré.

    Returns:
        DataFrame sans NaN.
    """
    # --- Colonnes numériques : imputation par la médiane ---
    numeric_cols = ['product_weight_g', 'product_photos_qty',
                    'total_price', 'total_freight', 'payment_value',
                    'payment_installments', 'item_count']

    for col in numeric_cols:
        if col in df.columns:
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                df[col] = df[col].fillna(median_val)
                print(f"  ✅ {col}: {n_missing} NaN → médiane ({median_val:.2f})")

    # --- Colonnes catégorielles : imputation par 'unknown' ---
    cat_cols = ['product_category_name_english', 'payment_type',
                'customer_state', 'customer_city']

    for col in cat_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                df[col] = df[col].fillna('unknown')
                print(f"  ✅ {col}: {n_missing} NaN → 'unknown'")

    return df


# ============================================================
# ÉTAPE 2 — FEATURE ENGINEERING
# ============================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Crée des variables explicatives disponibles avant la livraison.

        Features créées :
            - freight_ratio        : part des frais de port dans le prix total
            - price_per_item       : prix moyen par article commandé
            - purchase_month       : mois d'achat (saisonnalité)
            - purchase_day_of_week : jour de semaine d'achat

    Args:
        df: DataFrame après nettoyage.

    Returns:
        DataFrame enrichi avec les nouvelles features.
    """
    # --- Ratio frais de port : freight / prix total ---
    # On évite la division par zéro avec np.where
    df['freight_ratio'] = np.where(
        df['total_price'] > 0,
        df['total_freight'] / df['total_price'],
        0
    )

    # --- Prix moyen par article ---
    df['price_per_item'] = np.where(
        df['item_count'] > 0,
        df['total_price'] / df['item_count'],
        0
    )

    # --- Mois d'achat (saisonnalité) ---
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month

    # --- Jour de la semaine d'achat (0=Lundi, 6=Dimanche) ---
    df['purchase_day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek

    print(
        "  ✅ 4 nouvelles features créées : "
        "freight_ratio, price_per_item, "
        "purchase_month, purchase_day_of_week"
    )

    return df


def create_return_refund_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée la variable cible binaire 'is_return_refund_risk'.

        Définition (proxy pré-livraison sur Olist) :
            - 1 si la commande est annulée/non disponible
            - 0 sinon

        Cette définition évite l'usage de signaux post-livraison (ex: review_score)
        pour rester cohérent avec une prédiction avant livraison.

    Args:
        df: DataFrame nettoyé.

    Returns:
        DataFrame avec la colonne is_return_refund_risk (0 ou 1).
    """
    status_risk = df['order_status'].isin(['canceled', 'unavailable'])
    df['is_return_refund_risk'] = status_risk.astype(int)

    # Afficher la distribution des classes
    counts = df['is_return_refund_risk'].value_counts()
    total = len(df)
    print(f"\n  Distribution de la variable cible (is_return_refund_risk) :")
    print(f"    Risque élevé (1) : {counts.get(1, 0):>7,} ({counts.get(1, 0)/total*100:.1f}%)")
    print(f"    Risque faible(0) : {counts.get(0, 0):>7,} ({counts.get(0, 0)/total*100:.1f}%)")

    return df


def create_target(df: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    """
    Alias de compatibilité arrière.
    Le paramètre threshold est ignoré.
    """
    return create_return_refund_target(df)


# ============================================================
# ÉTAPE 3 — ENCODAGE & SÉLECTION DES FEATURES
# ============================================================

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les variables catégorielles en valeurs numériques
    pour les algorithmes de Machine Learning.

    Stratégie :
      - payment_type : LabelEncoder (peu de modalités)
      - customer_state : LabelEncoder (27 états brésiliens)
      - product_category : LabelEncoder (trop de modalités pour OneHot)

    Args:
        df: DataFrame avec les features créées.

    Returns:
        DataFrame avec les colonnes catégorielles encodées.
    """
    le = LabelEncoder()

    categorical_cols = ['payment_type', 'customer_state', 'product_category_name_english']

    for col in categorical_cols:
        if col in df.columns:
            # Convertir en string d'abord pour éviter les erreurs de type
            df[col] = df[col].astype(str)
            df[col + '_encoded'] = le.fit_transform(df[col])
            print(f"  ✅ {col} encodé → {col}_encoded ({df[col].nunique()} modalités)")

    return df


def select_features(df: pd.DataFrame) -> tuple:
    """
    Sélectionne les colonnes finales utilisées pour l'entraînement des modèles.

        Features retenues :
            - Variables temporelles d'achat (mois, jour)
      - Variables financières (prix, frais de port, ratio)
      - Variables produit (poids, photos)
      - Variables catégorielles encodées

    Args:
        df: DataFrame entièrement préparé.

    Returns:
        tuple: (X, y) où X est le DataFrame des features, y la série cible.
    """
    feature_columns = [
        # Features temporelles d'achat
        'purchase_month',
        'purchase_day_of_week',

        # Features financières
        'total_price',
        'total_freight',
        'freight_ratio',
        'price_per_item',
        'payment_value',
        'payment_installments',

        # Features commande
        'item_count',

        # Features produit
        'product_weight_g',
        'product_photos_qty',

        # Features catégorielles encodées
        'payment_type_encoded',
        'customer_state_encoded',
        'product_category_name_english_encoded',
    ]

    # Ne conserver que les colonnes effectivement présentes dans le DataFrame
    available_features = [col for col in feature_columns if col in df.columns]
    missing = set(feature_columns) - set(available_features)
    if missing:
        print(f"  ⚠️  Features absentes (ignorées) : {missing}")

    X = df[available_features].copy()
    y = df['is_return_refund_risk'].copy()

    # Supprimer les lignes avec des NaN résiduels dans les features
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    print(f"\n  ✅ {len(available_features)} features sélectionnées")
    print(f"  ✅ Dataset final : {X.shape[0]:,} lignes × {X.shape[1]} colonnes")
    return X, y


# ============================================================
# ÉTAPE 4 — SPLIT TRAIN / TEST
# ============================================================

def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """
    Divise les données en ensembles d'entraînement et de test.

    On utilise stratify=y pour maintenir la même proportion de classes
    dans les deux ensembles (important avec des classes déséquilibrées).

    Args:
        X: DataFrame des features.
        y: Série de la variable cible.
        test_size: Proportion du jeu de test (défaut: 20%).
        random_state: Graine aléatoire pour la reproductibilité.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintient la proportion des classes
    )

    print(f"\n  ✅ Split Train/Test ({int((1-test_size)*100)}/{int(test_size*100)}) :")
    print(f"     X_train : {X_train.shape[0]:,} lignes")
    print(f"     X_test  : {X_test.shape[0]:,} lignes")

    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> tuple:
    """
    Normalise les features numériques avec StandardScaler.
    (moyenne = 0, écart-type = 1)

    IMPORTANT : le scaler est entraîné UNIQUEMENT sur X_train
    pour éviter le data leakage (fuite d'information depuis le test set).

    Args:
        X_train: Features d'entraînement.
        X_test: Features de test.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
               Le scaler peut être réutilisé pour de nouvelles données.
    """
    scaler = StandardScaler()

    # fit_transform sur train : apprend ET transforme
    X_train_scaled = scaler.fit_transform(X_train)

    # transform uniquement sur test : applique sans réapprendre
    X_test_scaled = scaler.transform(X_test)

    print(f"  ✅ StandardScaler appliqué (fit sur train uniquement)")
    return X_train_scaled, X_test_scaled, scaler


# ============================================================
# PIPELINE COMPLET
# ============================================================

def full_preprocessing_pipeline(df: pd.DataFrame) -> tuple:
    """
    Exécute toutes les étapes de preprocessing dans le bon ordre.

    Usage dans un notebook :
        from src.preprocessing import full_preprocessing_pipeline
        X_train, X_test, y_train, y_test, scaler = full_preprocessing_pipeline(df)

    Args:
        df: DataFrame maître brut (retourné par load_data()).

    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("=" * 55)
    print("PIPELINE DE PREPROCESSING")
    print("=" * 55)

    print("\n📅 Conversion des dates...")
    df = convert_dates(df)

    print("\n🔍 Filtrage des commandes valides...")
    df = filter_valid_orders(df)

    print("\n🧹 Gestion des valeurs manquantes...")
    df = handle_missing_values(df)

    print("\n⚙️  Feature Engineering...")
    df = create_features(df)
    df = create_return_refund_target(df)

    print("\n🔡 Encodage des variables catégorielles...")
    df = encode_categorical(df)

    print("\n🎯 Sélection des features...")
    X, y = select_features(df)

    print("\n✂️  Split Train / Test...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n📏 Normalisation (StandardScaler)...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("\n" + "=" * 55)
    print("✅ PREPROCESSING TERMINÉ")
    print("=" * 55)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
