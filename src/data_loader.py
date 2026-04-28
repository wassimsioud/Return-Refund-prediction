# ============================================================
# src/data_loader.py
# ============================================================
# Responsabilité : charger les 9 fichiers CSV du dataset Olist
# et les fusionner en un seul DataFrame prêt pour l'analyse.
# ============================================================

import os
import pandas as pd


# Chemin vers le dossier data/ (relatif à la racine du projet)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MASTER_FILENAME = 'master.csv'


def load_raw_tables(data_dir: str = DATA_DIR) -> dict:
    """
    Charge les 9 fichiers CSV du dataset Olist dans un dictionnaire de DataFrames.

    Args:
        data_dir (str): Chemin vers le dossier contenant les fichiers CSV.

    Returns:
        dict: {nom_table: DataFrame}
    """
    # Noms des fichiers attendus dans le dossier data/
    files = {
        'orders':      'olist_orders_dataset.csv',
        'items':       'olist_order_items_dataset.csv',
        'payments':    'olist_order_payments_dataset.csv',
        'reviews':     'olist_order_reviews_dataset.csv',
        'customers':   'olist_customers_dataset.csv',
        'sellers':     'olist_sellers_dataset.csv',
        'products':    'olist_products_dataset.csv',
        'geolocation': 'olist_geolocation_dataset.csv',
        'translation': 'product_category_name_translation.csv',
    }

    tables = {}
    for name, filename in files.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Fichier manquant : {path}\n"
                f"Consulte data/download_instructions.md pour télécharger le dataset."
            )
        tables[name] = pd.read_csv(path)
        print(f"  ✅ {name:15s} — {tables[name].shape[0]:>7,} lignes | {tables[name].shape[1]} colonnes")

    return tables


def build_master_dataframe(tables: dict) -> pd.DataFrame:
    """
    Fusionne les tables du dataset Olist en un seul DataFrame "maître"
    contenant toutes les informations nécessaires au projet ML.

    Schéma de jointure :
        orders
          ├── customers    (sur customer_id)
          ├── items        (sur order_id)  → agrégé par order_id
          ├── payments     (sur order_id)  → agrégé par order_id
          ├── reviews      (sur order_id)  → on garde la dernière review
          └── products     (sur product_id, via items)
                └── translation (sur product_category_name)

    Args:
        tables (dict): Dictionnaire retourné par load_raw_tables().

    Returns:
        pd.DataFrame: DataFrame fusionné (~100 000 lignes).
    """

    # ------------------------------------------------------------------
    # 1. TABLE DE BASE : orders
    # ------------------------------------------------------------------
    df = tables['orders'].copy()

    # ------------------------------------------------------------------
    # 2. FUSION avec customers (infos de localisation du client)
    #    Clé : customer_id
    # ------------------------------------------------------------------
    customers = tables['customers'][['customer_id', 'customer_city', 'customer_state']].copy()
    df = df.merge(customers, on='customer_id', how='left')

    # ------------------------------------------------------------------
    # 3. FUSION avec reviews (note et commentaires)
    #    On garde UNE review par commande (la plus récente en cas de doublon)
    #    Clé : order_id
    # ------------------------------------------------------------------
    reviews = tables['reviews'].copy()
    # Garder la review la plus récente par commande
    reviews = reviews.sort_values('review_creation_date', ascending=False)
    reviews = reviews.drop_duplicates(subset='order_id', keep='first')
    reviews = reviews[['order_id', 'review_score', 'review_comment_message']]
    df = df.merge(reviews, on='order_id', how='left')

    # ------------------------------------------------------------------
    # 4. FUSION avec payments (montant total et mode de paiement)
    #    On agrège par order_id car une commande peut avoir plusieurs paiements
    #    Clé : order_id
    # ------------------------------------------------------------------
    payments = tables['payments'].copy()
    # Agrégation : somme du montant total, mode de paiement le plus utilisé
    payments_agg = payments.groupby('order_id').agg(
        payment_value=('payment_value', 'sum'),
        payment_installments=('payment_installments', 'max'),
        payment_type=('payment_type', lambda x: x.mode()[0])  # mode = valeur la plus fréquente
    ).reset_index()
    df = df.merge(payments_agg, on='order_id', how='left')

    # ------------------------------------------------------------------
    # 5. FUSION avec items (prix, frais de port, nombre d'articles)
    #    On agrège par order_id
    #    Clé : order_id
    # ------------------------------------------------------------------
    items = tables['items'].copy()
    items_agg = items.groupby('order_id').agg(
        item_count=('order_item_id', 'count'),         # nombre d'articles
        total_price=('price', 'sum'),                   # prix total HT
        total_freight=('freight_value', 'sum'),         # total frais de port
        seller_id=('seller_id', 'first')               # premier vendeur (simplification)
    ).reset_index()
    df = df.merge(items_agg, on='order_id', how='left')

    # ------------------------------------------------------------------
    # 6. FUSION avec products (catégorie du produit principal)
    #    On prend le premier produit de la commande (via items)
    #    Clé : product_id
    # ------------------------------------------------------------------
    # Récupère le product_id du premier article de chaque commande
    first_item = items[['order_id', 'product_id']].drop_duplicates(subset='order_id', keep='first')

    products = tables['products'][['product_id', 'product_category_name',
                                    'product_weight_g', 'product_photos_qty']].copy()

    # Traduction des catégories en anglais
    translation = tables['translation'].copy()
    products = products.merge(translation, on='product_category_name', how='left')

    # Fusion : items → products
    first_item = first_item.merge(products, on='product_id', how='left')
    df = df.merge(first_item[['order_id', 'product_category_name_english',
                               'product_weight_g', 'product_photos_qty']],
                  on='order_id', how='left')

    print(f"\n✅ DataFrame maître construit : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    return df


def save_master_dataframe(df: pd.DataFrame, data_dir: str = DATA_DIR) -> str:
    """
    Sauvegarde le DataFrame maître dans le dossier data/ sous le nom master.csv.

    Args:
        df: DataFrame maître à sauvegarder.
        data_dir: Dossier cible pour la sauvegarde.

    Returns:
        str: Chemin du fichier master.csv créé.
    """
    output_path = os.path.join(data_dir, MASTER_FILENAME)
    df.to_csv(output_path, index=False)
    print(f"💾 Fichier maître enregistré : {output_path}")
    return output_path


def load_data(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Point d'entrée principal : charge et fusionne toutes les tables.

    Usage dans un notebook :
        from src.data_loader import load_data
        df = load_data()

    Returns:
        pd.DataFrame: DataFrame maître prêt pour l'EDA et le preprocessing.
    """
    print("📂 Chargement des fichiers CSV...")
    tables = load_raw_tables(data_dir)
    print("\n🔗 Fusion des tables...")
    df = build_master_dataframe(tables)
    save_master_dataframe(df, data_dir=data_dir)
    return df