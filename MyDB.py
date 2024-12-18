import os
from werkzeug.utils import secure_filename
import json
import pandas as pd
import shutil

class MyDB:
    _save_directory = ''
    _tag_file_name = "tag_file.json"
    _symbol_list_file_name = 'symbol_list.csv'
    _user_id = ''
    _user_path = ''
    _tag_file_path = ''
    _symbol_list_path = ''
    _symbol_list_columns = ['Symbol Name', 'Overnight Margin', 'Markets']


    def __init__(self, user_id) -> None:
        self._user_id = user_id
        self._save_directory = 'users'
        self._user_path = os.path.join(self._save_directory, user_id)
        self._tag_file_path = os.path.join(self._user_path, self._tag_file_name)
        self._symbol_list_path = os.path.join(self._user_path, self._symbol_list_file_name)

    # If a file with the same name exists, it is replaced by the new one
    def insert_file(self, report) -> None:
        file_name = secure_filename(report.name)
        report_file_path = os.path.join(self._user_path, file_name)
        
        if os.path.exists(report_file_path): os.remove(report_file_path) # Delete the previously uploaded file
        with open(report_file_path, 'wb') as file: file.write(report.read())
        # Load existing data if file exists
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

        # Load existing data if file exists
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
                        print('Report not found')

    def get_numeber_report(self) -> int:
        with open(self._tag_file_path, 'r') as file:
            tag_file_content = json.load(file)
            return len(tag_file_content)
        
    def get_symbol_list(self):
        return pd.read_csv(self._symbol_list_path, header=None, names=self._symbol_list_columns, dtype={self._symbol_list_columns[0]:str, self._symbol_list_columns[1]: int, self._symbol_list_columns[2]: str})
    
    def save_symbol_list(self, symbol_df):
        symbol_df = symbol_df.fillna({
            self._symbol_list_columns[0]: '',
            self._symbol_list_columns[1]: 0,  
            self._symbol_list_columns[2]: ''
        })
        symbol_df.to_csv(self._symbol_list_path, header=False, index=False)

    # REPO MANAGER, STRATEGYPRO, COMPARATOR
    def get_df_report(self, report_name):
        report_path = os.path.join(self._user_path, report_name)
        df = pd.read_csv(report_path, header=None)
        return df
    
    ### FOLIO MANAGER
    def get_df_report2(self, report_name):
        report_path = os.path.join(self._user_path, report_name)
        df = pd.read_csv(report_path, header=None, nrows=1)
        return df

    