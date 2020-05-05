from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time

exec_string = '''function ClickConnect() {
  console.log('Working')
  document
    .querySelector('#top-toolbar > colab-connect-button')
    .shadowRoot.querySelector('#connect')
    .click()
}

setInterval(ClickConnect, 60000)'''


options = Options()
# chrome_options.add_argument("--start-fullscreen")
options.add_argument("--user-data-dir=C:\\Users\\max\AppData\\Local\\Google\\Chrome\\User Data\\Profile 2")
driver = webdriver.Chrome(executable_path=("C:\\Users\\max\\Desktop\\chromedriver.exe"), options=options)
driver.get('https://colab.research.google.com/drive/17NU7sFVL5rs9qve4zl8Ych3FEyKQUmXO')
driver.switch_to.new_window('tab')
driver.get("https://colab.research.google.com/drive/1Y68mYZox0F7VuVxmRfML81snLQbYZmOa")
driver.switch_to.new_window('tab')
driver.get("https://colab.research.google.com/drive/1hJQMbAUQpez8NOuJZSJgRY0kjNfOnjKk")
driver.switch_to.new_window('tab')
driver.get("https://colab.research.google.com/drive/1mw_VhpGxq5JMRzr_cXIfBQsyKci6Weiu")
driver.switch_to.new_window('tab')
driver.get("https://colab.research.google.com/drive/1dl_pX1hx9q3CPjzkWCv4jq2VRhq310_Z")
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
