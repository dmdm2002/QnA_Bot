import os
import pandas as pd


class DatabaseHandler:
    def __init__(self, rag_cfg):
        self.data_root = rag_cfg.data_root

        if os.path.isfile(f'{self.data_root}/enrollments.csv'):
            self.db = pd.read_csv(f'{self.data_root}/enrollments.csv')
        else:
            self.db = pd.DataFrame(columns=['id', 'name', 'status'])

    def check_new_data(self):
        # 해시셋으로 변환하여 O(1) 검색
        db_names = set(name.lower() for name in self.db['name']) # O(n) 생성
        raw_datas = [raw_data for raw_data in os.listdir(f'{self.data_root}/raw_data') 
                    if raw_data.endswith('.pdf')]
        
        update_datas = []
        for raw_data in raw_datas:
            if raw_data.lower() not in db_names:  # O(1) 검색
                print(f'-----New raw data: {raw_data}-----')
                update_datas.append(raw_data)
            else:
                print(f'-----Already exist raw data: {raw_data}-----')

        return update_datas

    def update_database(self, update_datas):
        update_datas = self.check_new_data()

        if not self.db.empty:
            max_id = self.db['id'].max()
        else:
            max_id = 0

        for update_data in update_datas:
            data_df = pd.DataFrame([[max_id, update_data, 'new']], columns=['id', 'name', 'status'])
            self.db = pd.concat([self.db, data_df], ignore_index=True)

            max_id += 1

        self.db.to_csv(f'{self.data_root}/enrollments.csv', index=False)
