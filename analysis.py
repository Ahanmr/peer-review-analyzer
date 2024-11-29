import openreview
import pandas as pd
import numpy as np
from tqdm import tqdm
import spacy
import sys 
from collections import defaultdict
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import os
from getpass import getpass
import csv

class OpenReviewAnalyzer:
    def __init__(self, baseurl='https://api2.openreview.net', username=None, password=None):
        """
        Initialize the OpenReview client with authentication using API v2
        
        Args:
            baseurl (str): OpenReview API v2 base URL
            username (str, optional): OpenReview username/email
            password (str, optional): OpenReview password
        """
        try:
            if username is None:
                username = os.getenv('OPENREVIEW_USERNAME')
            if password is None:
                password = os.getenv('OPENREVIEW_PASSWORD')
            
            if username is None:
                username = input("Enter your OpenReview email: ")
            if password is None:
                password = getpass("Enter your OpenReview password: ")
            
            self.client = openreview.api.OpenReviewClient(
                baseurl=baseurl,
                username=username,
                password=password
            )
            print("Successfully authenticated with OpenReview API v2!")
            
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("Downloading spaCy model...")
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load('en_core_web_sm')
                
        except Exception as e:
            print(f"Authentication failed: {str(e)}")
            raise
        
    def list_available_venues(self):
        """
        Get all available venues from OpenReview
        
        Returns:
            list: List of venue IDs
        """
        try:
            venues = self.client.get_group(id='venues')
            if venues and hasattr(venues, 'members'):
                print("\nAvailable venues:")
                for venue in venues.members:
                    print(f"- {venue}")
                return venues.members
            else:
                print("No venues found or unexpected response format")
                return []
        except Exception as e:
            print(f"Error getting venues: {str(e)}")
            return []

    def get_conference_submissions(self, conference_id):
        """
        Fetch all submissions for a given conference using API V2 method
        """
        try:
            # Get venue group and handle missing content
            venue_group = self.client.get_group(conference_id)
            
            # Handle different possible submission invitation patterns
            submission_patterns = [
                f'{conference_id}/-/Submission',  # Common pattern
                f'{conference_id}/Submission',    # Alternative pattern
                f'{conference_id}/-/Paper'        # Legacy pattern
            ]
            
            # Try to get submission_name from venue content
            if hasattr(venue_group, 'content'):
                if isinstance(venue_group.content, dict):
                    submission_name = venue_group.content.get('submission_name', {}).get('value', 'Submission')
                    submission_patterns.insert(0, f'{conference_id}/-/{submission_name}')
            
            # Try each pattern until we find submissions
            submissions = []
            for pattern in submission_patterns:
                try:
                    submissions = self.client.get_all_notes(
                        invitation=pattern,
                        details='replies'
                    )
                    if submissions:
                        print(f"Found submissions using pattern: {pattern}")
                        break
                except Exception as e:
                    continue
                
            return submissions
            
        except Exception as e:
            print(f"Error getting submissions: {str(e)}")
            return []

    def get_reviews_for_conference(self, conference_id):
        """
        Fetch all reviews for a conference using API V2 method
        """
        try:
            # Get submissions first
            submissions = self.get_conference_submissions(conference_id)
            if not submissions:
                print("No submissions found")
                return []
            
            # Common review invitation patterns
            review_patterns = [
                'Official_Review',
                'Review',
                'review'
            ]
            
            # Try to get review_name from venue content
            try:
                venue_group = self.client.get_group(conference_id)
                if hasattr(venue_group, 'content') and isinstance(venue_group.content, dict):
                    review_name = venue_group.content.get('review_name', {}).get('value')
                    if review_name:
                        review_patterns.insert(0, review_name)
            except:
                pass
            
            # Extract reviews from submission replies
            reviews = []
            for submission in submissions:
                if not hasattr(submission.details, 'replies'):
                    continue
                
                for reply in submission.details['replies']:
                    # Check if any review pattern matches the invitation
                    if any(pattern in reply.get('invitation', '') for pattern in review_patterns):
                        try:
                            review = openreview.api.Note.from_json(reply)
                            reviews.append(review)
                        except:
                            continue
            
            if not reviews:
                print("No reviews found")
            else:
                print(f"Found {len(reviews)} reviews")
            
            return reviews
            
        except Exception as e:
            print(f"Error getting reviews: {str(e)}")
            return []

    def extract_review_features(self, review):
        """
        Extract relevant features from a review with better error handling
        
        Args:
            review: Review object
            
        Returns:
            dict: Dictionary containing extracted features
        """
        try:
            # Extract basic metadata with safer content access
            features = {
                'review_id': review.id,
                'paper_id': review.forum,
                'timestamp': datetime.fromtimestamp(review.tmdate/1000),
            }
            
            # Safely extract review content with different possible field names
            if isinstance(review.content, dict):
                # Try different possible field names for rating
                rating = None
                for rating_field in ['rating', 'recommendation', 'score', 'evaluation']:
                    rating = review.content.get(rating_field)
                    if rating is not None:
                        break
                features['rating'] = rating

                # Try different possible field names for confidence
                confidence = None
                for conf_field in ['confidence', 'reviewer_confidence']:
                    confidence = review.content.get(conf_field)
                    if confidence is not None:
                        break
                features['confidence'] = confidence

                # Try different possible field names for review text
                review_text = ''
                for text_field in ['review', 'comment', 'assessment']:
                    text = review.content.get(text_field, '')
                    if text:
                        review_text = text
                        break
                features['review_text'] = review_text
            else:
                print(f"Warning: Unexpected content structure for review {review.id}")
                features.update({
                    'rating': None,
                    'confidence': None,
                    'review_text': ''
                })

            # Convert rating to numeric if possible
            if features['rating']:
                try:
                    # Handle different rating formats
                    if isinstance(features['rating'], str):
                        # Try to extract numeric value from string
                        numeric_part = ''.join(filter(lambda x: x.isdigit() or x == '.', features['rating']))
                        features['rating'] = float(numeric_part) if numeric_part else None
                    else:
                        features['rating'] = float(features['rating'])
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert rating to numeric for review {review.id}")
                    features['rating'] = None

            # Only perform NLP analysis if we have review text
            if features['review_text']:
                # Add semantic analysis
                doc = self.nlp(features['review_text'][:1000000])  # Limit text length to avoid memory issues
                
                # Sentiment analysis
                blob = TextBlob(features['review_text'])
                features['sentiment_polarity'] = blob.sentiment.polarity
                features['sentiment_subjectivity'] = blob.sentiment.subjectivity
                
                # Extract key phrases and entities
                features['key_phrases'] = [chunk.text for chunk in doc.noun_chunks]
                features['named_entities'] = [ent.text for ent in doc.ents]
            else:
                features.update({
                    'sentiment_polarity': None,
                    'sentiment_subjectivity': None,
                    'key_phrases': [],
                    'named_entities': []
                })

            return features
            
        except Exception as e:
            print(f"Warning: Error processing review {review.id}: {str(e)}")
            # Return a default feature set if processing fails
            return {
                'review_id': review.id,
                'paper_id': review.forum,
                'timestamp': datetime.fromtimestamp(review.tmdate/1000),
                'rating': None,
                'confidence': None,
                'review_text': '',
                'sentiment_polarity': None,
                'sentiment_subjectivity': None,
                'key_phrases': [],
                'named_entities': []
            }

    def create_review_dataset(self, conference_id):
        """
        Create a structured dataset of reviews for a conference
        
        Args:
            conference_id (str): The conference ID
            
        Returns:
            pd.DataFrame: Structured dataset of reviews
        """
        all_reviews_data = []
        
        # Get all reviews directly
        reviews = self.get_reviews_for_conference(conference_id)
        print(f"Found {len(reviews)} reviews")
        
        for review in tqdm(reviews, desc="Processing reviews"):
            review_features = self.extract_review_features(review)
            all_reviews_data.append(review_features)
        
        return pd.DataFrame(all_reviews_data)

    def analyze_reviews(self, df):
        """
        Perform analysis on the review dataset with better handling of missing values
        
        Args:
            df (pd.DataFrame): Review dataset
            
        Returns:
            dict: Analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis['total_reviews'] = len(df)
        
        # Handle numeric calculations safely
        rating_series = pd.to_numeric(df['rating'], errors='coerce')
        confidence_series = pd.to_numeric(df['confidence'], errors='coerce')
        
        analysis['avg_rating'] = rating_series.mean()
        analysis['rating_std'] = rating_series.std()
        analysis['rating_counts'] = rating_series.value_counts().to_dict()
        
        analysis['avg_confidence'] = confidence_series.mean()
        analysis['confidence_std'] = confidence_series.std()
        
        # Sentiment distribution (excluding None values)
        sentiment_df = df[df['sentiment_polarity'].notna()]
        analysis['sentiment_stats'] = {
            'mean_polarity': sentiment_df['sentiment_polarity'].mean() if not sentiment_df.empty else None,
            'mean_subjectivity': sentiment_df['sentiment_subjectivity'].mean() if not sentiment_df.empty else None
        }
        
        # Time analysis
        df['month'] = df['timestamp'].dt.month
        analysis['reviews_by_month'] = df['month'].value_counts().sort_index().to_dict()
        
        return analysis

    def visualize_analysis(self, df, analysis_results):
        """
        Create visualizations for the analysis
        
        Args:
            df (pd.DataFrame): Review dataset
            analysis_results (dict): Analysis results
        """
        # Set up the plotting style
        plt.style.use('seaborn')
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Rating distribution
        sns.histplot(data=df, x='rating', ax=axes[0,0])
        axes[0,0].set_title('Distribution of Ratings')
        
        # Confidence distribution
        sns.histplot(data=df, x='confidence', ax=axes[0,1])
        axes[0,1].set_title('Distribution of Reviewer Confidence')
        
        # Sentiment polarity vs. rating
        sns.scatterplot(data=df, x='sentiment_polarity', y='rating', ax=axes[1,0])
        axes[1,0].set_title('Sentiment Polarity vs. Rating')
        
        # Reviews over time
        time_data = pd.DataFrame.from_dict(
            analysis_results['reviews_by_month'], 
            orient='index',
            columns=['count']
        )
        time_data.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Reviews by Month')
        
        plt.tight_layout()
        plt.show()

    def export_reviews_to_csv_simple(self, conference_id, output_file=None):
        """
        A simplified and robust method to export reviews to CSV
        
        Args:
            conference_id (str): The conference ID
            output_file (str, optional): Output CSV filename
        """
        try:
            # Get all reviews
            reviews = self.get_reviews_for_conference(conference_id)
            
            if not reviews:
                print("No reviews found to export")
                return
            
            if output_file is None:
                output_file = f'reviews_{conference_id.replace("/", "_")}.csv'
            
            # Define basic fields that should exist in most reviews
            basic_fields = ['id', 'forum', 'signature', 'tmdate']
            
            # Get all possible content fields from all reviews
            content_fields = set()
            for review in reviews:
                if hasattr(review, 'content') and isinstance(review.content, dict):
                    content_fields.update(review.content.keys())
            
            # Prepare the header
            header = basic_fields + list(content_fields)
            
            # Write to CSV
            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)
                
                for review in reviews:
                    row = []
                    
                    # Add basic fields
                    for field in basic_fields:
                        value = getattr(review, field, '')
                        if field == 'tmdate':
                            # Convert timestamp to readable date
                            try:
                                value = datetime.fromtimestamp(value/1000).strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                value = ''
                        row.append(value)
                    
                    # Add content fields
                    for field in content_fields:
                        try:
                            content = review.content.get(field, '')
                            # Handle different content formats
                            if isinstance(content, dict):
                                if 'value' in content:
                                    value = content['value']
                                else:
                                    value = str(content)
                            else:
                                value = content
                            row.append(value)
                        except:
                            row.append('')
                    
                    writer.writerow(row)
            
            print(f"Successfully exported {len(reviews)} reviews to {output_file}")
            
            # Also create a more detailed version with submission titles
            try:
                # Get submissions to map forums to titles
                submissions = self.get_all_submissions(conference_id)
                forum_to_title = {sub.id: sub.content.get('title', {}).get('value', '') 
                                if isinstance(sub.content.get('title'), dict) 
                                else sub.content.get('title', '') 
                                for sub in submissions}
                
                detailed_output = f'detailed_{output_file}'
                with open(output_file, 'r', newline='', encoding='utf-8') as infile, \
                     open(detailed_output, 'w', newline='', encoding='utf-8') as outfile:
                    reader = csv.reader(infile)
                    writer = csv.writer(outfile)
                    
                    # Write header with additional fields
                    header = next(reader)
                    writer.writerow(['paper_title'] + header)
                    
                    # Write data with paper titles
                    for row in reader:
                        forum = row[header.index('forum')]
                        title = forum_to_title.get(forum, '')
                        writer.writerow([title] + row)
                
                print(f"Also created detailed export with paper titles: {detailed_output}")
                
            except Exception as e:
                print(f"Could not create detailed version with paper titles: {str(e)}")
            
        except Exception as e:
            print(f"Error during export: {str(e)}")
            
            # Last resort: try to dump raw review data
            try:
                emergency_output = f'emergency_{output_file}'
                with open(emergency_output, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(['review_id', 'forum', 'raw_content'])
                    
                    for review in reviews:
                        writer.writerow([
                            getattr(review, 'id', ''),
                            getattr(review, 'forum', ''),
                            str(getattr(review, 'content', {}))
                        ])
                print(f"Created emergency export with raw data: {emergency_output}")
            except:
                print("Failed to create even emergency export")

    def get_metareviews(self, conference_id):
        """
        Get all metareviews for a conference
        
        Args:
            conference_id (str): The conference ID
            
        Returns:
            list: List of metareview objects
        """
        try:
            # Get all submissions with their direct replies
            submissions = self.client.get_all_notes(
                invitation=f"{conference_id}/-/Submission",
                details='directReplies'
            )
            
            # Extract metareviews from submission replies
            metareviews = []
            for submission in submissions:
                metareviews.extend([
                    reply for reply in submission.details["directReplies"] 
                    if reply["invitation"].endswith("Meta_Review")
                ])
                
            return metareviews
        
        except Exception as e:
            print(f"Error getting metareviews: {str(e)}")
            return []

    def get_all_submissions(self, conference_id, status='all'):
        """
        Get all submissions for a conference with specified status
        """
        try:
            submissions = []
            
            # Try different submission patterns
            patterns = [
                f'{conference_id}/-/Submission',
                f'{conference_id}/Submission',
                f'{conference_id}/-/Paper'
            ]
            
            # Try to get submission_name from venue content
            try:
                venue_group = self.client.get_group(conference_id)
                if hasattr(venue_group, 'content') and isinstance(venue_group.content, dict):
                    submission_name = venue_group.content.get('submission_name', {}).get('value')
                    if submission_name:
                        patterns.insert(0, f'{conference_id}/-/{submission_name}')
            except:
                pass

            # Try each pattern
            for pattern in patterns:
                try:
                    if status == 'all':
                        submissions = self.client.get_all_notes(invitation=pattern)
                    else:
                        # Try to get venue IDs from group content or use fallback patterns
                        venue_ids = []
                        if status == 'accepted':
                            venue_ids = [conference_id]
                        elif status == 'active':
                            venue_ids = [f'{conference_id}/Under_Review']
                        elif status == 'withdrawn':
                            venue_ids = [f'{conference_id}/Withdrawn']
                        elif status == 'desk-rejected':
                            venue_ids = [f'{conference_id}/Desk_Rejected']
                        
                        for venue_id in venue_ids:
                            try:
                                these_submissions = self.client.get_all_notes(
                                    content={'venueid': venue_id}
                                )
                                submissions.extend(these_submissions)
                            except:
                                continue
                
                    if submissions:
                        print(f"Found submissions using pattern: {pattern}")
                        break
                except:
                    continue
                
            return submissions
            
        except Exception as e:
            print(f"Error getting submissions: {str(e)}")
            return []

def main():
    try:
        analyzer = OpenReviewAnalyzer()
        
        # First, list all available venues
        print("\nListing all available venues...")
        venues = analyzer.list_available_venues()
        
        if venues:
            # Let user choose a venue
            print("\nAvailable venues:")
            for i, venue in enumerate(venues):
                print(f"{i+1}. {venue}")
            
            choice = input("\nEnter the number of the venue to analyze (or press Enter for ICLR.cc/2024/Conference): ")
            
            if choice.strip():
                conference_id = venues[int(choice)-1]
            else:
                conference_id = 'ICLR.cc/2023/Conference'
            
            print(f"\nAnalyzing conference: {conference_id}")
            
            # Export reviews (this will work even if other operations fail)
            print("\nExporting reviews to CSV...")
            analyzer.export_reviews_to_csv_simple(conference_id)
            
            # Continue with other analysis if possible...
            try:
                # Get all submissions
                print("\nGetting submissions...")
                all_submissions = analyzer.get_all_submissions(conference_id)
                print(f"Total submissions: {len(all_submissions)}")
                
                # Get and analyze reviews
                print("\nAnalyzing reviews...")
                review_df = analyzer.create_review_dataset(conference_id)
                
                if not review_df.empty:
                    analysis_results = analyzer.analyze_reviews(review_df)
                    
                    # Visualize results
                    print("\nGenerating visualizations...")
                    analyzer.visualize_analysis(review_df, analysis_results)
            except Exception as e:
                print(f"Additional analysis failed: {str(e)}")
                print("But reviews were exported to CSV successfully")
                
        else:
            print("No venues available")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
