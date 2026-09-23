{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d8fa8395",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "82eddf07",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading: c:\\Users\\Lenovo\\Documents\\projects\\context-aware-neural-recommendation-engine\\data\\raw\\articles.csv\\articles.csv\n",
      "articles: (105542, 25)\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "\n",
    "folder = Path.cwd().parent / \"data\" / \"raw\" / \"articles.csv\"\n",
    "\n",
    "if folder.is_dir():\n",
    "    file = [f for f in folder.rglob(\"*.csv\") if f.is_file()][0]\n",
    "else:\n",
    "    file = folder\n",
    "\n",
    "print(\"Loading:\", file)\n",
    "articles = pd.read_csv(file)\n",
    "print(\"articles:\", articles.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ff1e9012",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['t_dat', 'customer_id', 'article_id', 'price', 'sales_channel_id']\n",
      "transactions shape: (31788324, 5)\n"
     ]
    }
   ],
   "source": [
    "p = RAW / \"transactions_train.csv\" / \"transactions_train.csv\"\n",
    "\n",
    "# columns\n",
    "cols = pd.read_csv(p, nrows=5)\n",
    "print(cols.columns.tolist())\n",
    "\n",
    "# row count\n",
    "with open(p, \"rb\") as f:\n",
    "    n_rows = sum(1 for _ in f) - 1   # minus header\n",
    "\n",
    "print(\"transactions shape:\", (n_rows, cols.shape[1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "750fdcb6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C:\\Users\\lenovo\\Desktop\\customers.csv\\customers.csv\n",
      "(1371980, 7)\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "\n",
    "folder = Path(r\"C:\\Users\\lenovo\\Desktop\\customers.csv\")\n",
    "file = [f for f in folder.iterdir() if f.is_file()][0]\n",
    "print(file)\n",
    "\n",
    "customers = pd.read_csv(file)\n",
    "print(customers.shape)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "bc27edc8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "customers: (1371980, 7)\n",
      "articles: (105542, 25)\n",
      "transactions: (31788324, 5)\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "\n",
    "RAW = Path.cwd().parent / \"data\" / \"raw\"\n",
    "\n",
    "def get_csv_path(path):\n",
    "    p = Path(path)\n",
    "    if p.is_dir():\n",
    "        p = [f for f in p.rglob(\"*.csv\") if f.is_file()][0]\n",
    "    return p\n",
    "\n",
    "# Customers\n",
    "customers_path = get_csv_path(RAW / \"customers.csv\")\n",
    "customers = pd.read_csv(customers_path)\n",
    "print(\"customers:\", customers.shape)\n",
    "\n",
    "# Articles\n",
    "articles_path = get_csv_path(RAW / \"articles.csv\")\n",
    "articles = pd.read_csv(articles_path)\n",
    "print(\"articles:\", articles.shape)\n",
    "\n",
    "# Transactions (count rows without loading the whole file)\n",
    "transactions_path = get_csv_path(RAW / \"transactions_train.csv\")\n",
    "num_cols = pd.read_csv(transactions_path, nrows=5).shape[1]\n",
    "with open(transactions_path, \"rb\") as f:\n",
    "    num_rows = sum(1 for _ in f) - 1   # minus header row\n",
    "print(\"transactions:\", (num_rows, num_cols))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
