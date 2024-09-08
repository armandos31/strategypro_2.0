import os
from werkzeug.utils import secure_filename
import json
import pandas as pd
import shutil

class MyDB:
    _example_files_dir = 'file_esempio'
    _example_paramenter_file = 'opt_report.txt'
    _save_directory = ''
    _tag_file_name = "tag_file.json"
    _symbol_list_file_name = 'symbol_list.csv'
    _user_id = ''
    _user_path = ''
    _tag_file_path = ''
    _symbol_list_path = ''
    _symbol_list_columns = ['Symbol Name', 'Overnight Margin', 'Markets']

    def __init__(self, user_id, isGuest) -> None:
        self._user_id = user_id
        if isGuest:
            self._user_path = self._example_files_dir
        else:
            self._save_directory = 'users'
            self._user_path = os.path.join(self._save_directory, user_id)
        if not os.path.exists(self._user_path):
            os.makedirs(self._user_path)
        self._tag_file_path = os.path.join(self._user_path, self._tag_file_name)
        if not os.path.exists(self._tag_file_path):
            with open(self._tag_file_path, 'w') as file:
                empty_list = []
                json.dump(empty_list, file)
        self._symbol_list_path = os.path.join(self._user_path, self._symbol_list_file_name)
        if not os.path.exists(self._symbol_list_path):
            symbol_df = pd.DataFrame(columns=self._symbol_list_columns)
            symbol_df[self._symbol_list_columns[0]] = symbol_df[self._symbol_list_columns[0]].astype(str)
            symbol_df[self._symbol_list_columns[1]] = symbol_df[self._symbol_list_columns[1]].astype(int)
            symbol_df[self._symbol_list_columns[2]] = symbol_df[self._symbol_list_columns[2]].astype(str)
            symbol_df[self._symbol_list_columns[0]] = ["@ES"]
            symbol_df[self._symbol_list_columns[1]] = [2500]
            symbol_df[self._symbol_list_columns[2]] = ["index"]
            symbol_df.to_csv(self._symbol_list_path, header=False, index=False)



    # Se esiste un file con lo stesso nome, questo viene sostituito da quello nuovo
    def insert_file(self, report) -> None:
        file_name = secure_filename(report.name)
        report_file_path = os.path.join(self._user_path, file_name)
        
        if os.path.exists(report_file_path): os.remove(report_file_path) # Elimina il file precedentemente caricato
        with open(report_file_path, 'wb') as file: file.write(report.read())

        # Carica dati esistenti se il file esiste
        try:
            with open(self._tag_file_path, 'r') as file:
                tag_file_content = json.load(file)
        except FileNotFoundError:
            tag_file_content = []

        is_report_update = False
        for entry in tag_file_content:
            if entry['file_name'] == file_name:
                is_report_update = True
                break

        if not is_report_update:
            new_tag_file_element = [{'file_name': file_name, 'tags': []}]
            tag_file_content.extend(new_tag_file_element)
            with open(self._tag_file_path, 'w') as file:
                json.dump(tag_file_content, file, indent=2)

    def insert_portfolio(self, df_portfolio, file_name) -> None:        
        portfolio_file_path = os.path.join(self._user_path, file_name)
        df_portfolio.to_csv(portfolio_file_path, index=False)

        # Carica dati esistenti se il file esiste
        try:
            with open(self._tag_file_path, 'r') as file:
                tag_file_content = json.load(file)
        except FileNotFoundError:
            tag_file_content = []

        is_report_update = False
        for entry in tag_file_content:
            if entry['file_name'] == file_name:
                is_report_update = True
                break

        if not is_report_update:
            new_tag_file_element = [{'file_name': file_name, 'tags': []}]
            tag_file_content.extend(new_tag_file_element)
            with open(self._tag_file_path, 'w') as file:
                json.dump(tag_file_content, file, indent=2)

    
    def get_tag_file(self) -> list:
        try:
            with open(self._tag_file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            return []    
    
    def mod_tag(self, op, tags, df_reports) -> bool:
        try:
            with open(self._tag_file_path, 'r') as file:
                tag_file_content = json.load(file)
                for report in df_reports.itertuples(index=False):
                    report_name = report._0
                    for elemento in tag_file_content:
                        if elemento["file_name"] == report_name:
                            if op == 'Add Tags':
                                set_A = set(elemento["tags"])
                                set_B = set(tags)
                                set_A.update(set_B)
                                elemento["tags"] = list(set_A)
                            else:
                                set_A = set(elemento["tags"])
                                set_B = set(tags)
                                set_A.difference_update(set_B)
                                elemento["tags"] = list(set_A)
                            break
                with open(self._tag_file_path, 'w') as file:
                    json.dump(tag_file_content, file, indent=2)
            return True
        except FileNotFoundError:
            return False

    def delete_report(self, df_reports) -> bool:
            for report in df_reports.itertuples(index=False):
                report_name = report._0
                report_file_path = os.path.join(self._user_path, report_name)
                if os.path.exists(report_file_path):
                    try:
                        #print('Report trovato, lo elimino')
                        os.remove(report_file_path)
                        with open(self._tag_file_path, 'r') as file:
                            tag_file_content = json.load(file)
                            for elemento in tag_file_content:
                                if elemento["file_name"] == report_name:
                                    tag_file_content.remove(elemento)
                                    break
                            with open(self._tag_file_path, 'w') as file:
                                json.dump(tag_file_content, file, indent=2)
                    except OSError:
                        print('Report non trovato')

    def get_numeber_report(self) -> int:
        with open(self._tag_file_path, 'r') as file:
            tag_file_content = json.load(file)
            return len(tag_file_content)
        
    def get_symbol_list(self):
        return pd.read_csv(self._symbol_list_path, header=None, names=self._symbol_list_columns, dtype={self._symbol_list_columns[0]:str, self._symbol_list_columns[1]: int, self._symbol_list_columns[2]: str})
    
    def save_symbol_list(self, symbol_df):
        # symbol_df = symbol_df.dropna()
        symbol_df = symbol_df.fillna({
            self._symbol_list_columns[0]: '',
            self._symbol_list_columns[1]: 0,   # Riempi le celle vuote nella colonna di interi con 0
            self._symbol_list_columns[2]: ''
        })
        symbol_df.to_csv(self._symbol_list_path, header=False, index=False)

    def get_df_report(self, report_name):
        report_path = os.path.join(self._user_path, report_name)
        df = pd.read_csv(report_path, header=None)
        return df
    
    def get_example_paramenter_file(self):
        p_path = os.path.join(self._user_path, self._example_paramenter_file)
        df = pd.read_csv(p_path, encoding="utf-16", sep='\t')
        return df

    def get_df_report2(self, report_name):
        report_path = os.path.join(self._user_path, report_name)
        df = pd.read_csv(report_path, header=None, nrows=1)
        return df