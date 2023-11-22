<body>
<h1>Time Series Prediction using LSTM</h1>
    <p>This project focuses on predicting daily mean temperature using Long Short-Term Memory (LSTM) networks. The dataset used is the Daily Delhi Climate data.</p>
<p>Long Short-Term Memory (LSTM) model is a type of recurrent neural network (RNN) designed to handle the challenges of learning and remembering long-term dependencies in sequential data. LSTMs use a specialized architecture with memory cells and gates to selectively add, remove, and output information. This allows them to capture and retain important patterns and dependencies in time series data. Key components of LSTMs include the cell state (long-term memory), hidden state (short-term memory or output), and gates (input, forget, and output gates) that regulate the flow of information. LSTMs are particularly effective for tasks like time series prediction, natural language processing, and speech recognition where understanding and remembering sequences of information are crucial. They address the vanishing gradient problem, a common issue in traditional RNNs, making them well-suited for learning from sequences with long-term dependencies.

</p>
    <h2>Libraries</h2>
    <p><code>numpy</code> and <code>pandas</code> for data manipulation</p>
    <p><code>matplotlib</code> for data visualization</p>
    <p><code>keras</code> for building and training the LSTM model</p>
    <p><code>sklearn</code> for data normalization</p>



<h2>Dataset</h2>
    <p>The dataset is Daily Delhi Climate data, loaded from the CSV file <code>DailyDelhiClimateTrain.csv</code>. It contains columns such as date, mean temperature, humidity, wind speed, and mean pressure.</p>
    <pre><code>
data_dir = 'Data/DailyDelhiClimateTrain.csv'
df = pd.read_csv(data_dir)
    </code></pre>

<h2>Contributing</h2>
    <p>Feel free to contribute to this project. If you have suggestions or find issues, please open an <a
            href="https://github.com/your-username/your-repository/issues">issue</a>.</p>

<h2>License</h2>
    <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

</body>
