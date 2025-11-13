import datetime
import os
from typing import Callable, Optional
import pandas as pd
from sklearn import preprocessing
import numpy as np
import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset
)

pd.set_option('display.max_columns', None)

class AMLtoGraph(InMemoryDataset):

    def __init__(self, root: str, dataset_type_size: str, edge_window_size: int = 10, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        self.dataset_type_size = dataset_type_size
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        if self.dataset_type_size == 'HISMALL':
            return 'HI-Small_Trans.csv'
        elif self.dataset_type_size == 'HIMEDIUM':
            return 'HI-Medium_Trans.csv'
        elif self.dataset_type_size == 'LISMALL':
            return 'LI-Small_Trans.csv'
        elif self.dataset_type_size == 'LIMEDIUM':
            return 'LI-Medium_Trans.csv'
        else:
            raise ValueError("dataset_type_size must be one of 'HISMALL', 'HIMEDIUM', 'LISMALL', 'LIMEDIUM'")

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.dataset_type_size}.pt'

    @property
    def num_nodes(self) -> int:
        return self._data.edge_index.max().item() + 1

    def df_label_encoder(self, df, columns):
        le = preprocessing.LabelEncoder()
        for i in columns:
            df[i] = le.fit_transform(df[i].astype(str))
        return df


    def preprocess(self, df):
        df = self.df_label_encoder(df,['Payment Format', 'Payment Currency', 'Receiving Currency'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Timestamp'] = df['Timestamp'].apply(lambda x: x.value)
        df['Timestamp'] = (df['Timestamp']-df['Timestamp'].min())/(df['Timestamp'].max()-df['Timestamp'].min())

        df['Account'] = df['From Bank'].astype(str) + '_' + df['Account']
        df['Account.1'] = df['To Bank'].astype(str) + '_' + df['Account.1']
        df = df.sort_values(by=['Account'])
        receiving_df = df[['Account.1', 'Amount Received', 'Receiving Currency']]
        paying_df = df[['Account', 'Amount Paid', 'Payment Currency']]
        receiving_df = receiving_df.rename({'Account.1': 'Account'}, axis=1)
        currency_ls = sorted(df['Receiving Currency'].unique())

        return df, receiving_df, paying_df, currency_ls

    def get_all_account(self, df):
        ldf = df[['Account', 'From Bank', 'Timestamp']]
        rdf = df[['Account.1', 'To Bank', 'Timestamp']]
        suspicious = df[df['Is Laundering'] == 1]
        s1 = suspicious[['Account', 'Is Laundering']]
        s2 = suspicious[['Account.1', 'Is Laundering']].rename({'Account.1': 'Account'}, axis=1)
        suspicious = pd.concat([s1, s2], join='outer').drop_duplicates()

        ldf = ldf.rename({'From Bank': 'Bank'}, axis=1)
        rdf = rdf.rename({'Account.1': 'Account', 'To Bank': 'Bank'}, axis=1)
        accounts = pd.concat([ldf, rdf], join='outer', ignore_index=True)

        accounts = (
            accounts.sort_values(['Account', 'Timestamp'])
            .groupby('Account', as_index=False)
            .agg({'Bank': 'last', 'Timestamp': 'max'})
        )
        accounts = accounts.rename({'Timestamp': 'latest_timestamp'}, axis=1)

        accounts['Is Laundering'] = 0
        accounts.set_index('Account', inplace=True)
        accounts.update(suspicious.set_index('Account'))
        accounts['Is Laundering'] = 1 - accounts['Is Laundering'] # Flip labels: 0=illicit, 1=licit
        return accounts.reset_index()

    
    def paid_currency_aggregate(self, currency_ls, paying_df, accounts):
        for i in currency_ls:
            temp = paying_df[paying_df['Payment Currency'] == i]
            means = temp.groupby('Account')['Amount Paid'].mean()
            accounts['avg paid ' + str(i)] = accounts['Account'].map(means)
        return accounts

    def received_currency_aggregate(self, currency_ls, receiving_df, accounts):
        for i in currency_ls:
            temp = receiving_df[receiving_df['Receiving Currency'] == i]
            means = temp.groupby('Account')['Amount Received'].mean()
            accounts['avg received ' + str(i)] = accounts['Account'].map(means)
        accounts = accounts.fillna(0)
        return accounts

    def transaction_feature_aggregate(self, transactions_df):
        outgoing = transactions_df.groupby('Account').agg(
            out_tx_count=('Amount Paid', 'count'),
            out_amount_paid_sum=('Amount Paid', 'sum'),
            out_amount_paid_mean=('Amount Paid', 'mean'),
            out_timestamp_mean=('Timestamp', 'mean')
        )

        incoming = transactions_df.groupby('Account.1').agg(
            in_tx_count=('Amount Received', 'count'),
            in_amount_received_sum=('Amount Received', 'sum'),
            in_amount_received_mean=('Amount Received', 'mean'),
            in_timestamp_mean=('Timestamp', 'mean')
        )
        incoming.index = incoming.index.rename('Account')
        outgoing.index = outgoing.index.rename('Account')

        stats = outgoing.join(incoming, how='outer')
        stats = stats.fillna(0)
        stats['net_amount_sum'] = stats['out_amount_paid_sum'] - stats['in_amount_received_sum']
        stats['net_tx_count'] = stats['out_tx_count'] - stats['in_tx_count']
        return stats.reset_index()

    def get_edge_df(self, accounts, df):
        accounts = accounts.reset_index(drop=True)
        accounts['ID'] = accounts.index
        mapping_dict = dict(zip(accounts['Account'], accounts['ID']))
        df = df.copy()
        df['From'] = df['Account'].map(mapping_dict)
        df['To'] = df['Account.1'].map(mapping_dict)

        edge_index = torch.stack(
            [torch.from_numpy(df['From'].values), torch.from_numpy(df['To'].values)],
            dim=0
        ).to(torch.long)

        edge_features = df.drop(
            ['Account', 'Account.1', 'From Bank', 'To Bank', 'Is Laundering', 'From', 'To'],
            axis=1
        )
        edge_attr = torch.from_numpy(edge_features.values).to(torch.float)

        reverse_edge_index = edge_index.flip(0)
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr.clone()], dim=0)
        return edge_attr, edge_index

    def get_node_attr(self, currency_ls, paying_df,receiving_df, accounts, transactions_df):
        node_df = self.paid_currency_aggregate(currency_ls, paying_df, accounts)
        node_df = self.received_currency_aggregate(currency_ls, receiving_df, node_df)
        txn_stats = self.transaction_feature_aggregate(transactions_df)
        node_df = node_df.merge(txn_stats, on='Account', how='left')
        node_df = node_df.fillna(0)
        node_label = torch.from_numpy(node_df['Is Laundering'].values).to(torch.long)
        feature_df = node_df.drop(['Account', 'Is Laundering'], axis=1)
        feature_df = self.df_label_encoder(feature_df, ['Bank'])
        node_attr = torch.from_numpy(feature_df.values).to(torch.float)
        return node_attr, node_label

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        df, receiving_df, paying_df, currency_ls = self.preprocess(df)
        accounts = self.get_all_account(df)
        node_attr, node_label = self.get_node_attr(currency_ls, paying_df,receiving_df, accounts, df)
        edge_attr, edge_index = self.get_edge_df(accounts, df)

        data = Data(x=node_attr,
                    edge_index=edge_index,
                    y=node_label,
                    edge_attr=edge_attr
                    )
        
        data_list = [data] 
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
