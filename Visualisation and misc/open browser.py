from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
def open_browser(in_list):
    exec_string = '''function ClickConnect() {
      console.log('Working')
      document
        .querySelector('#top-toolbar > colab-connect-button')
        .shadowRoot.querySelector('#connect')
        .click()
    }
    
    setInterval(ClickConnect, 60000)'''


    options = Options()
    options.add_argument("--user-data-dir=C:\\Users\\max\AppData\\Local\\Google\\Chrome\\User Data\\Profile 2")
    driver = webdriver.Chrome(executable_path=("C:\\Users\\max\\Desktop\\chromedriver.exe"), options=options)
    driver.get(in_list[0])
    for i in range(1, len(in_list)):
        driver.switch_to.new_window('tab')
        driver.get(in_list[i])

    for handle in driver.window_handles:
        driver.switch_to.window(handle)
        webdriver.ActionChains(driver).key_down(Keys.CONTROL).send_keys(Keys.F9).perform()
        time.sleep(0.1)
        webdriver.ActionChains(driver).key_up(Keys.CONTROL).perform()
        time.sleep(0.2)

    for handle in driver.window_handles:
        driver.switch_to.window(handle)
        time.sleep(0.5)
        driver.execute_script(exec_string)

procedural_list = ['https://colab.research.google.com/drive/17NU7sFVL5rs9qve4zl8Ych3FEyKQUmXO',
                   "https://colab.research.google.com/drive/1Y68mYZox0F7VuVxmRfML81snLQbYZmOa",
                   "https://colab.research.google.com/drive/1hJQMbAUQpez8NOuJZSJgRY0kjNfOnjKk",
                   "https://colab.research.google.com/drive/1mw_VhpGxq5JMRzr_cXIfBQsyKci6Weiu",
                   "https://colab.research.google.com/drive/1dl_pX1hx9q3CPjzkWCv4jq2VRhq310_Z"]
multitask_list = ["https://colab.research.google.com/drive/1ahXlcJ_7KEFu3KUTlk0UKPvfowcPLd5j",
                  "https://colab.research.google.com/drive/1d2cAvxr8VkgVyegN2wUZQ6Mn5lySdwKL",
                  "https://colab.research.google.com/drive/1XAP0vvt6eXxif7mMh1IM-2CLZdAZnitD",
                  "https://colab.research.google.com/drive/1I1jzn7XK2DQ5Du7OI3LYN5Iw8lNMYgRW",
                  "https://colab.research.google.com/drive/1j4ZNzjpqTkATIdba4hgcCwDbBeJn-dwu"]

open_browser(multitask_list)
