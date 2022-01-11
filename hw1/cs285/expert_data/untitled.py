import pickle


load_file=open("expert_data_Ant-v2.pkl","rb")
load_game_data=pickle.load(load_file)
print(load_game_data)
load_file.close()