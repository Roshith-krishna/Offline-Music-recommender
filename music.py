import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os
dataset_path = "dataset.csv"

class MusicEngine:

    def __init__(self,dataset_path):
        print("dataset loading...")
        self.df = pd.read_csv(dataset_path)
        self.df = self.df.dropna() 
        self.df = self.df.drop_duplicates(subset=['track_name','artists'])
        self.df = self.df.reset_index(drop=True)
        self.liked_file = "liked_songs.csv"
        self.disliked_file = "disliked_songs.csv"
        self.feature_cols = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo']
        self.scaler = StandardScaler()
        self.feature_matrix = self.scaler.fit_transform(self.df[self.feature_cols])
        self.model = NearestNeighbors(n_neighbors=10,algorithm='brute',metric='cosine')
        self.model.fit(self.feature_matrix)
        self.load_user_history()

    def load_user_history(self):
        if not os.path.exists(self.liked_file):
            pd.DataFrame(columns=self.df.columns).to_csv(self.liked_file,index=False)
        if not os.path.exists(self.disliked_file):
            pd.DataFrame(columns=self.df.columns).to_csv(self.disliked_file,index=False)
        self.liked_df = pd.read_csv(self.liked_file)
        self.disliked_df = pd.read_csv(self.disliked_file)

    def save_interaction(self,song_row,liked=True):
        song_df = pd.DataFrame([song_row])
        if liked:
            if song_row['track_id'] not in self.liked_df['track_id'].values:
                song_df.to_csv(self.liked_file,mode='a',header=False,index=False)
                print(f"saved to likes:{song_row['track_name']}")
        else:
            if song_row['track_id'] not in self.disliked_df['track_id'].values:
                song_df.to_csv(self.disliked_file,mode='a',header=False,index=False)
                print(f"saved to dislikes:{song_row['track_name']}")
        self.load_user_history()  

    def get_user_vectors(self):
        if len(self.liked_df)==0:
            return None
        liked_features = self.scaler.transform(self.liked_df[self.feature_cols])
        positive_vector= np.mean(liked_features,axis=0)
        if len(self.disliked_df) > 0:
            disliked_features = self.scaler.transform(self.disliked_df[self.feature_cols])
            negative_vector = np.mean(disliked_features,axis=0)
            final_vector = positive_vector-(0.3*negative_vector)
        else:
            final_vector = positive_vector
        return final_vector.reshape(1, -1)
    
    def find_song_index(self, song_name):
        result=self.df[self.df['track_name'].str.contains(song_name,case=False,na=False)]
        if len(result)==0:
            return None
        return result.index[0]
    
    
    def recommend_similar(self, song_name):
        idx = self.find_song_index(song_name)
        if idx is None:
            print(f" Could not find song: '{song_name}'")
            return []
        input_vector = self.feature_matrix[idx].reshape(1, -1)
        distances, indices = self.model.kneighbors(input_vector, n_neighbors=6)
        recommendations = []
        for i in range(1, len(indices[0])):
            row_id = indices[0][i]
            song = self.df.iloc[row_id]
            recommendations.append(song)   
        return recommendations

    def recommend_random_personalized(self):
        user_vector=self.get_user_vectors()
        if user_vector is None:
            return None
        distance , indices = self.model.kneighbors(user_vector,n_neighbors=50)
        valid_indices=[]
        for idx in indices[0]:
            track_id=self.df.iloc[idx]['track_id']
            is_liked = track_id in self.liked_df['track_id'].values
            is_disliked= track_id in self.disliked_df['track_id'].values
            if not is_liked and not is_disliked:
                valid_indices.append(idx)
        if not valid_indices:
            print("you've rated everything nearby")
            return None
        chosen_idx=np.random.choice(valid_indices)
        return self.df.iloc[chosen_idx]

if __name__ == "__main__":

    if not os.path.exists(dataset_path):
        print(" Error: 'spotify_tracks.csv' not found.")
        print("Please download the Spotify Tracks Dataset from Kaggle and place it in this folder.")
    
    else:
        engine = MusicEngine(dataset_path)
        while True:
            print("\n" + "="*30)
            print("  SPOTIFY LOCAL RECOMMENDATIONS  ")
            print("="*30)
            print("\n1.  Search & Find Similar Songs")
            print("2.  Discover New Music (Personalized)")
            print("3.  Save & Exit")
            choice = input("\nSelect an option (1-3): ")
            if choice == '1':
                query = input("Enter a song name to search: ")
                recs = engine.recommend_similar(query)
                if recs:
                    print(f"\nSongs similar to your search:")
                    for i, song in enumerate(recs):
                        print(f"{i+1}. {song['track_name']} - {song['artists']}")
                    
                    print("\nWhich ones do you LIKE? (Type numbers separated by space, e.g., '1 3')")
                    print("Press Enter to skip.")
                    selection = input("> ")
                    
                    if selection.strip():
                        for num in selection.split():
                            if num.isdigit():
                                idx = int(num) - 1 
                                if 0 <= idx < len(recs):
                                    engine.save_interaction(recs[idx], liked=True)
            elif choice == '2':
                print("\n🔮 Analyzing your taste profile...")
                song = engine.recommend_random_personalized() 
                if song is not None:
                    print("\n" + "-"*30)
                    print(f"I RECOMMEND:  {song['track_name']}")
                    print(f"ARTIST:       {song['artists']}")
                    print("-"*30)
                    feedback = input("\nDid you like this? (y = Yes / n = No / s = Skip): ").lower()
                    if feedback == 'y':
                        engine.save_interaction(song, liked=True)
                    elif feedback == 'n':
                        engine.save_interaction(song, liked=False)
                    else:
                        print("Skipped.")
            elif choice == '3':
                print("Goodbye!")
                break
            else:
                print("Invalid option. Try again.")

          
