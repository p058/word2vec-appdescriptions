import pandas as pd


def get_descriptions(x):
    import urllib2
    from bs4 import BeautifulSoup
    try:

        try:

            response = urllib2.urlopen("https://itunes.apple.com/us/app/id" + x)
            soup = BeautifulSoup(response.read(), 'html.parser')
            desc = soup.find(itemprop='description').get_text()
            genre = soup.find(itemprop='applicationCategory')
            if genre is not None:

                genre = genre.get_text()

            else:

                genre = 'Others'

            return (x, desc, genre)
        except:

            response = urllib2.urlopen("https://play.google.com/store/apps/details?id=" + x)
            soup = BeautifulSoup(response.read(), 'html.parser')
            desc = soup.find(itemprop='description').get_text()
            genre = soup.find(itemprop='genre')

            if genre is not None:

                genre = genre.get_text()

            else:

                genre = 'Others'

            return (x, desc, genre)

    except:

        return (x, "N", "N")

if __name__=='__main__':
    exampleapplist=['com.aws.android','com.weather.Weather']
    appdescs=[]
    for app in exampleapplist:
        appdescs.append(get_descriptions(app))
    appdescs_pd = pd.DataFrame(appdescs, columns=['App Bundle ID', 'Description', 'Category'])
    appdescs_pd_filtered = appdescs_pd[appdescs_pd['App Bundle ID'] != "N"]
    print appdescs_pd_filtered
