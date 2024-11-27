import pandas as pd
import requests
import time
from tqdm import tqdm
import os
from typing import Dict, List, Optional

def create_conference_url(conference: str, year: int, offset: int = 0, limit: int = 20) -> str:
    """create the api url for a given conference & year"""
    base_url = 'https://api.openreview.net/notes'
    invitation = f'{conference}.cc%2F{year}%2FConference%2F-%2FBlind_Submission'
    return f'{base_url}?invitation={invitation}&offset={offset}&limit={limit}'

def fetch_all_papers(conference: str, year: int) -> pd.DataFrame:
    """fetchs all papers"""
    papers = []
    offset = 0
    limit = 100
    pbar = tqdm(desc=f"Fetching papers for {conference} {year}", unit=" papers")
    while True:
        url = create_conference_url(conference, year, offset, limit)
        response = requests.get(url).json()
        if not response['notes']:
            break
        batch_size = len(response['notes'])
        papers.extend(response['notes'])
        offset += limit
        pbar.update(batch_size)
    pbar.close()
    return pd.DataFrame(papers)

def fetch_forum_content(forum_ids: List[str], conference: str, year: int) -> List[Dict]:
    """fetch forum content for a list of forum IDs"""
    forum_content = []
    for forum_id in tqdm(forum_ids, desc=f"Fetching {conference} {year} forum content", unit=" forums"):
        response = requests.get(f'https://api.openreview.net/notes?forum={forum_id}&trash=true').json()
        forum_content.append(response)
        time.sleep(0.3) # rate limits 
    return forum_content

def process_reviews(df: pd.DataFrame, conference: str, year: int) -> pd.DataFrame:
    """Process the raw dataframe to extract reviews and decisions."""
    print(f"Processing reviews for {conference} {year}...")
    #extract paper information
    df['title'] = df.content.apply(lambda x: x['title'])
    df['authors'] = df.content.apply(lambda x: x['authors'])
    df['decision_raw'] = df.forumContent.apply(lambda x: [n['content']['decision'] 
                                                         for n in x['notes']
                                                         if 'decision' in n['content']][0]
                                                         if any('decision' in n['content'] 
                                                               for n in x['notes']) else 'Unknown')

    #extract reviews
    only_reviews_df = pd.concat(df.forumContent.apply(lambda c: pd.DataFrame([
        {'review': n['content']['review'],
         'rating': n['content'].get('rating', 'Unknown'),
         'confidence': n['content'].get('confidence', 'Unknown'),
         'forum': n['forum']}
        for n in c['notes']
        if 'content' in n and 'review' in n['content']
    ])).tolist())

    #merge reviews with paper information
    reviews_df = pd.merge(df[['title', 'authors', 'decision_raw', 'forum']], 
                         only_reviews_df, on='forum')

    #categorize decisions and ratings
    reviews_df['decision'] = reviews_df['decision_raw'].apply(
        lambda x: 'Reject' if x == 'Reject'
        else ('Accept' if x.startswith('Accept')
              else ('Workshop' if x.startswith('Workshop') else 'Unknown')))

    reviews_df['rating_bin'] = reviews_df['rating'].apply(
        lambda x: 'Unknown' if x == 'Unknown' else (
            lambda s: 'Negative' if s < 5
            else ('Positive' if s > 6 else 'Neutral')
        )(int(str(x).split(':')[0].strip())))

    reviews_df['category'] = reviews_df['decision'] + ', ' + reviews_df['rating_bin']
    
    return reviews_df

def scrape_conference(conference: str, year: int) -> None:
    """Main function to scrape conference data and save it."""
    print(f"\nScraping {conference} {year}...")
    os.makedirs('datasets', exist_ok=True)
    df = fetch_all_papers(conference, year)
    if df.empty:
        print(f"No data found for {conference} {year}")
        return
    df['forumContent'] = fetch_forum_content(df.forum.tolist(), conference, year)
    reviews_df = process_reviews(df, conference, year)
    output_file = f'datasets/{conference.lower()}_{year}_reviews.csv.bz2'
    reviews_df.to_csv(output_file, index=False, compression='bz2')
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    conferences = ['NeurIPS', 'ICLR']
    years = range(2020, 2024)
    total_tasks = len(conferences) * len(years)
    with tqdm(total=total_tasks, desc="Overall Progress", unit=" conference-year") as pbar:
        for conference in conferences:
            for year in years:
                try:
                    scrape_conference(conference, year)
                except Exception as e:
                    print(f"Error scraping {conference} {year}: {str(e)}")
                finally:
                    pbar.update(1)