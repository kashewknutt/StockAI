# Stock Chatbot Application

![Stock Chatbot](https://your-image-link.com/stock-chatbot-banner.png)

## Description

Welcome to the Stock Chatbot Application, a sophisticated Django-based web application designed to provide users with real-time stock information, predictions, and conversational assistance. This chatbot leverages machine learning models and web scraping techniques to deliver detailed insights into stock prices and market trends. Users can interact with the chatbot through natural language queries, making it a user-friendly tool for both novice and experienced investors.

## Features

- **Real-time Stock Data**: Get up-to-date stock prices and market information.
- **Stock Predictions**: Utilize trained models to forecast stock performance.
- **Conversational AI**: Engage with a chatbot that can understand and respond to natural language queries about stocks.
- **Responsive Design**: A clean, modern, and responsive interface for optimal user experience.
- **Cost-Free Hosting**: Deploy the application using free hosting solutions.

## Technologies Used

- **Python**: The core programming language.
- **Django**: Web framework for the backend.
- **Transformers**: For NLP and chatbot functionality.
- **Selenium & BeautifulSoup**: For web scraping stock data.
- **Joblib**: For loading and using machine learning models.
- **Bootstrap**: For responsive and modern UI design.
- **jQuery & AJAX**: For real-time interactions and dynamic content updates.

## Setup Instructions

### Prerequisites

- **Python 3.7+**
- **pip** (Python package manager)
- **Django 3.2+**
- **Virtual Environment** (recommended for isolating dependencies)
- **ChromeDriver** (for web scraping)

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/kashewknutt/stock-chatbot.git
   cd stock-chatbot
   ```
2. **Create and Activate a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations**:

   ```bash
   python manage.py migrate
   ```

5. **Run the server**:

   ```bash
   python manage.py runserver
   ```
The application will be accessible at http://127.0.0.1:8000/

### Usage

1. **Access the Homepage**:

   Navigate to `http://localhost:8000/` to view the landing page of the Stock Chatbot application.

2. **Interact with the Chatbot**:

   Go to `http://localhost:8000/chatbot/` to start interacting with the chatbot. You can ask questions about current stock prices, predictions, and general stock-related information.

3. **Contact Page**:

   Use the contact form at `http://localhost:8000/contact/` to send messages or inquiries directly through the website.

### Deployment

To deploy this application, consider using free hosting solutions like Heroku or Vercel. Follow the platform-specific instructions for deploying a Django application.

1. **Prepare for Deployment**:
   - Set up your project environment variables in the hosting platform's dashboard.
   - Ensure all dependencies are listed in `requirements.txt`.

2. **Deploy on Heroku**:
   - Install the Heroku CLI and log in.
   - Create a new Heroku app:
     ```bash
     heroku create your-app-name
     ```
   - Push your code to Heroku:
     ```bash
     git push heroku main
     ```
   - Set up environment variables on Heroku through the Heroku dashboard.

3. **Deploy on Vercel**:
   - Sign up for a Vercel account and link it to your GitHub repository.
   - Import your project in the Vercel dashboard and follow the prompts to deploy.

### Project Structure

```plaintext
stock-chatbot/
│
├── chatbot/
│   ├── templates/
│   │   ├── chatbot.html              # HTML template for the chatbot page
│   │   ├── index.html                # Landing page
│   │   └── contact.html              # Contact page
│   ├── views.py                      # Django views
│   ├── models.py                     # Data models (if any)
│   ├── urls.py                       # URL routing for the chatbot app
│   ├── companyDatabase.py            # Company name to stock symbol mapping
│   ├── credentials.py                # Secure handling of API keys (not committed to version control)
│   ├── train_model.py                # Scripts for training the stock prediction model
│   └── ...
├── contact/
│   ├── templates/
│   │   ├── contact_success.html      # HTML template for the contact page
│   │   └── contact.html              # Contact page
│   ├── views.py                      # Django views
│   ├── models.py                     # Data models (if any)
│   ├── urls.py                       # URL routing for the chatbot app
│   └── ...
├── home/
│   ├── templates/
│   │   └── index.html                # Contact page
│   ├── views.py                      # Django views
│   ├── models.py                     # Data models (if any)
│   ├── urls.py                       # URL routing for the chatbot app
│   └── ...
│
├── stockAI/
│   ├── templates/
│   │   └── base.html                 # Base page
│   ├── static/
│   │   └── css/
│   │       └── styles.css            # Custom CSS for styling
│   ├── credentials.py                # Secure handling of email details (not committed to version control)
│   ├── settings.py                   # Django project settings
│   ├── urls.py                       # URL routing for the entire project
│   ├── wsgi.py                       # WSGI application
│   ├── asgi.py                       # ASGI application
│   └── ...
│
├── manage.py                         # Django management script
├── requirements.txt                  # List of project dependencies
└── README.md                         # Project documentation
```

### Contributing

Contributions are welcome and encouraged! To contribute to this project, please follow these steps:

1. **Fork the Repository**:
   - Click the "Fork" button on the top right corner of the GitHub page to create a copy of the repository on your account.

2. **Clone the Repository**:
   - Clone your forked repository to your local machine using:
     ```bash
     git clone https://github.com/kashewknutt/stockAI.git
     ```

3. **Create a New Branch**:
   - Create a new branch for your feature or bug fix:
     ```bash
     git checkout -b feature/kashewknutt
     ```

4. **Make Your Changes**:
   - Make the necessary changes to the codebase, ensuring you follow the project’s style guidelines and include appropriate tests.

5. **Commit Your Changes**:
   - Commit your changes with a descriptive message:
     ```bash
     git commit -m "Add feature: Your descriptive message"
     ```

6. **Push Your Changes**:
   - Push your changes to your forked repository:
     ```bash
     git push origin feature/your-feature-name
     ```

7. **Create a Pull Request**:
   - Go to the original repository and create a pull request from your forked repository's branch.

8. **Review Process**:
   - Your pull request will be reviewed, and any necessary changes will be requested. Once approved, your changes will be merged into the main branch.

Please ensure that your contributions do not break existing functionality and that they align with the project's goals. Feel free to open an issue if you have any questions or need further guidance on your contribution.

### License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software, provided that you include the original license and copyright notice in any copy of the software that you distribute.

**MIT License**

```plaintext
Copyright (c) 2024 [Your Name or Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

For more details, please refer to the [LICENSE](LICENSE) file.
