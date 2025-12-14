# 🎵 My DIY Music Recommender

I got a bit tired of streaming services telling me what to listen to, so I built my own music brain that runs 100% locally on my laptop. No cloud, no tracking, just Python and some math.

It’s basically a private DJ that lives in your terminal and reads a massive spreadsheet of songs to find you new bangers.

## 🤔 What is this?
It's a Python script that takes a huge list of Spotify tracks (with data like "Danceability", "Energy", "Tempo") and uses the **K-Nearest Neighbors** algorithm to find songs that are mathematically similar to the ones you like.

It starts out knowing nothing about you. You have to teach it.
- **You Like a song:** It moves your "User Profile" closer to that song's vibe.
- **You Dislike a song:** It pushes your profile away from that vibe.

## 🛠️ How to Run It

### 1. The Ingredients
You need Python installed, obviously. Then grab the libraries:
``` bash

pip install -r requirements.txt

```
### 2. The Data

Since this runs offline, it needs a database.
 1. Download the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) from Kaggle.

 2. Unzip it and rename the file to dataset.csv.

  3.Put it in the same folder as the script.

### 3. Let's Go

Run the script:
```bash

python music.py

```
## 🎮 How to Train Your Dragon (I mean, AI)

When you first run it, the AI is "dumb." It has no idea who you are.

    Start with Option 1 (Search): Search for 3-4 songs you really love .

    "Like" them: This builds your initial profile.

    Switch to Option 2 (Discover): Now the magic happens. It will suggest a song based on your history.

    Give Feedback:

        If it recommends a bad song, hit "No". The AI actually learns from this and adjusts your math vector to avoid songs like that in the future.

        If it's good, hit "Yes", and it gets smarter.

## ⚠️ A Small Limitation

Music is an ocean, but this dataset is just a bucket. 
Since this runs offline without the live Spotify API, the database "only" contains about 114,000 songs. It’s a lot, but your favorite niche song might be missing. If the search comes up empty, that's why!


## 🤓 The Nerd Stuff

Under the hood, this uses scikit-learn. It turns every song into a coordinate in a multi-dimensional space (Energy vs. Acousticness vs. Valence, etc.).

When you ask for a recommendation, it calculates the Cosine Similarity between your "Average Taste" and every other song in the database, then picks the closest matches.

Built by a bored CSE student who likes privacy and music.
