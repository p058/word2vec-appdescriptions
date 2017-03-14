import pandas as pd

def get_descriptions(x):
    import time
    import urllib2
    from bs4 import BeautifulSoup
    try:

        try:

            response = urllib2.urlopen("https://itunes.apple.com/us/app/id" + x)
            soup = BeautifulSoup(response.read(), 'html.parser')
            desc = soup.find(itemprop='description').get_text()
            time.sleep(2)
            return (x, desc)
        except:

            response = urllib2.urlopen("https://play.google.com/store/apps/details?id=" + x)
            soup = BeautifulSoup(response.read(), 'html.parser')
            desc = soup.find(itemprop='description').get_text()
            time.sleep(2)
            return (x, desc)

    except:

        return (x, "N")

if __name__=='__main__':
    exampleapplist=['com.aws.android','com.weather.Weather']
    appdescs=[]
    for app in exampleapplist:
        appdescs.append(get_descriptions(app))
    appdescs_pd = pd.DataFrame(appdescs, columns=['App Bundle ID', 'Description'])
    appdescs_pd_filtered = appdescs_pd[appdescs_pd['App Bundle ID'] != "N"]
    print appdescs_pd
