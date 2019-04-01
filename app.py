from flask import Flask, render_template, redirect, request, flash, url_for
import io
import csv
from werkzeug import secure_filename
import pandas as pd
import numpy as np
from random import choice, sample
import random

from random import choice, sample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

user_influencer = pd.read_csv('user_influencer.csv', index_col = 0)
subset_item = pd.read_csv('subset_item.csv')
influencers = ['weworewhat', 'chiaraferragni', 'blaireadiebee', 'somethingnavy', 'hannahbronfman',
              'nicolettemason', 'manrepeller', 'jordynwoods', 'seaofshoes', 'ariellecharnas']

def filter_item(list_items = [], list_brand = [], list_occasion = [], list_category = [], list_price = []):
    df = subset_item[~subset_item['category_name'].isin(['Beauty', 'Kids'])]
    if list_items != []:
        df = df[~df['product_id'].isin(list_items)]
    if list_brand != []:
        df = df[df['brand_id'].isin(list_brand)]
    if list_occasion != []:
        list_occasion = "|".join(list_occasion)
        df = df[df['occasion'].str.contains(list_occasion, na = False)]
    if list_category != []:
        df = df[df['category_name'].isin(list_category)]
    if list_price != []:
        if list_price == ['a']:
            df = df[(df['paid_price'] < 25)]
        elif list_price == ['c']:
            df = df[df['paid_price'] >= 70]
        else:
            df = df[(df['paid_price'] >= 25) & (df['paid_price'] < 70)]
    return df


def most_popular(n_items, list_items = [], list_brand = [], list_occasion = [], list_category = [], list_price = []):
    item = filter_item(list_items, list_brand, list_occasion, list_category, list_price)
    popular_item = item.groupby(["product_id", "category_name"])[["has_product"]].sum().reset_index()
    popular_item = popular_item.sort_values(by = "has_product", ascending = False)[:n_items * 5]
    popular_item = random.sample(popular_item['product_id'].tolist(), n_items)
    recommendation = fill_table(item, popular_item, n_items)
    return recommendation


influencers = ['weworewhat', 'chiaraferragni', 'blaireadiebee', 'somethingnavy', 'hannahbronfman',
               'nicolettemason', 'manrepeller', 'jordynwoods', 'seaofshoes', 'ariellecharnas']

def recommendation_influencer_aff(list_influencer, n_item, list_items = [], list_brand = [], list_occasion = [], list_category = [], list_price = []):
    influencer = pd.DataFrame(data = 0, columns = influencers, index = ['new user'])
    for influ in list_influencer:
        influencer[influ] = 1
    similarity_user = similarity(user_influencer.append(pd.DataFrame(influencer)))
    similar_users = similarity_user['new user'].sort_values(ascending = False)[1:]
    if (similar_users[0] == 0):
        recommendation = most_popular(n_item, list_items, list_brand, list_occasion, list_category)
    else:
        similar_users = similar_users[similar_users > 0.5].index
        item = filter_item(list_items, list_brand, list_occasion, list_category, list_price)
        similar_subset = item[item['user_id'].isin(similar_users)]
        similar_subset = similar_subset.groupby(["product_id", "category_name"])[["has_product"]].sum().reset_index()
        similar_subset = similar_subset[similar_subset['has_product'] > 0].sort_values(by = "has_product", ascending = False)
        if len(similar_subset) <= n_item:
            n_item = len(similar_subset)
            similar_subset = similar_subset['product_id']
        else:
            similar_subset = random.sample(similar_subset['product_id'].tolist(), n_item)
        recommendation = fill_table(item, similar_subset, n_item)
    return recommendation


def recommend_new_user(list_influencer = [], n_recommendations = 5, list_brand = [], list_occasion = [], list_category =[], list_price = []):
    list_items = []
    if list_influencer == []:
        return most_popular(n_recommendations, list_items, list_brand, list_occasion, list_category, list_price)
    else:
        return recommendation_influencer_aff(list_influencer, n_recommendations, list_items, list_brand, list_occasion, list_category, list_price)



userBrands = subset_item.groupby(['user_id','brand_id']).count()
userBrands = userBrands.reset_index().pivot(index='user_id', columns='brand_id', values = 'item_name_lower')
userBrands.fillna(0, inplace = True)
brandAffinity = np.ones(len(userBrands.columns)**2).reshape(len(userBrands.columns),len(userBrands.columns))*.05 + np.eye(len(userBrands.columns))
occasions = ['Beach','Work','School','Errands','Dinner','Formal event','Concert',
            'Night life','Cocktail parties','Brunch','Interview','Movies','Workout']

def recommend_user_exists(user_id, n_item = 5, occasion = [], category = [], list_price = []):
    uBrandScore = np.matmul(userBrands.loc[[user_id]],brandAffinity)
    brands = uBrandScore.T.sample(40, replace = True).index.tolist()
    item = subset_item[subset_item['user_id'] == user_id]['product_id'].unique().tolist()
    influencer = user_influencer.filter(like = user_id, axis = 0)
    influencer = [column for column in influencer.columns if influencer.loc[user_id][column] == 1]
    if influencer == []:
        item_to_recommend = most_popular(n_item, item, brands, occasion, category, list_price)
    else:
        item_to_recommend = recommendation_influencer_aff(influencer, n_item, item, brands, occasion, category, list_price)
    return item_to_recommend



def fill_table(df, product_id, n_items):
    recommendation = pd.DataFrame(data = 0, index = range(n_items), 
                                  columns = ['Item Name', 'Brand', 'Store', 
                                             'Category', 'Price', 'Color', 'Occasion'])
    for row, item in enumerate (product_id):
        try:
            recommendation['Item Name'][row] = df[df['product_id'] == item]['item_name'].mode()[0]
        except IndexError:
            recommendation['Item Name'][row] = 'No Item Name'
        try:
            recommendation['Brand'][row] = df[df['product_id'] == item]['parsed_brand_name'].mode()[0]
        except IndexError:
            recommendation['Brand'][row] = 'No Brand'
        try:
            recommendation['Store'][row] = df[df['product_id'] == item]['parsed_store_name'].mode()[0]
        except IndexError:
            recommendation['Store'][row] = 'No Store'
        try:
            recommendation['Category'][row] = df[df['product_id'] == item]['category_name'].mode()[0]
        except IndexError:
            recommendation['Category'][row] = 'No Category'
        try:
            recommendation['Price'][row] = df[df['product_id'] == item]['paid_price'].mean()
        except IndexError:
            recommendation['Price'][row] = 'No Price'
        try:
            recommendation['Color'][row] = df[df['product_id'] == item]['color_parsed'].mode()[0]
        except IndexError:
            recommendation['Color'][row] = 'No Color'
        try:
            recommendation['Occasion'][row] = df[df['product_id'] == item]['occasion'].mode()[0]
        except IndexError:
            recommendation['Occasion'][row] = 'No Occasion'
    return recommendation

def similarity(df):
    users_ids = df.index
    similarity = pd.DataFrame(data = cosine_similarity(df), 
                         index = users_ids, columns = users_ids)
    return similarity




app=Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")



influencer_selected=''
@app.route('/influencer/', methods=['GET', 'POST'])
def influencer():
    global influencer_selected
    for check in request.form:
        influencer_selected = request.form.getlist(check)
        return redirect('/occasion_new/')
    return render_template("influencer.html")

recommendation=''
occasion_selected=''
user_id = ''
@app.route('/occasion/', methods=['GET', 'POST'])
def occasion():
    global recommendation
    global occasion_selected
    for check in request.form:
        occasion_selected = request.form.getlist(check)
        print (occasion_selected)
        return redirect('/category/')
    global user_id
    if user_id == '':
        user_id = subset_item['user_id'].sample(1).values[0] 
    else:
        user_id = user_id
    recommendation=recommend_user_exists(user_id = user_id)
    return render_template("occasion.html", recommendation=recommendation, user_id = user_id)

category_selected=''
@app.route('/category/', methods=['GET', 'POST'])
def category():
    global category_selected
    for check in request.form:
        category_selected = request.form.getlist(check)
        return redirect('/price/')
    recommendation=recommend_user_exists(user_id = user_id, occasion=occasion_selected)
    return render_template("category.html", recommendation=recommendation, user_id = user_id)

@app.route('/occasion_new/', methods=['GET', 'POST'])
def occasion_new():
    global occasion_selected
    for check in request.form:
        occasion_selected = request.form.getlist(check)
        return redirect('/category_new/')
    recommendation=recommend_new_user(list_influencer = influencer_selected)
    return render_template("occasion_new.html", recommendation=recommendation)



@app.route('/category_new/', methods=['GET', 'POST'])
def category_new():
    global category_selected
    for check in request.form:
        category_selected = request.form.getlist(check)
        return redirect('/price_new/')
    recommendation=recommend_new_user(list_influencer = influencer_selected, list_occasion=occasion_selected)
    return render_template("category_new.html", recommendation=recommendation)


@app.route('/price_new/', methods=['GET', 'POST'])
def price_new():
    global price_selected
    for check in request.form:
        price_selected = request.form.getlist(check)
        return redirect('/display_new/')
    recommendation=recommend_new_user(list_influencer = influencer_selected, list_occasion=occasion_selected, list_category =category_selected)
    return render_template("price_new.html", recommendation=recommendation)


price_selected=''
@app.route('/price/', methods=['GET', 'POST'])
def price():
    global price_selected
    for check in request.form:
        price_selected = request.form.getlist(check)
        return redirect('/display/')
    recommendation=recommend_user_exists(user_id = user_id, occasion=occasion_selected, category =category_selected)
    return render_template("price.html", recommendation=recommendation, user_id = user_id)


@app.route('/display_new/', methods=['GET', 'POST'])
def display_new():
    global category_selected
    for check in request.form:
        category_selected = request.form.getlist(check)
        return redirect('/')
    recommendation=recommend_new_user(list_influencer = influencer_selected, list_occasion=occasion_selected, list_category =category_selected, list_price = price_selected)
    return render_template("display_new.html", recommendation=recommendation)


@app.route('/display/', methods=['GET', 'POST'])
def display():
    recommendation=recommend_user_exists(user_id = user_id, occasion=occasion_selected, category =category_selected, list_price = price_selected)
    return render_template("display.html", recommendation=recommendation, user_id = user_id)

if __name__ == "__main__":
    app.run(debug=True)









