import React from 'react';
import axios from 'axios';

class App extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      generatedDog: null 
    }
  }

  generateDog = async () => {
    const res = await axios.get('/api/generate') 
    this.setState({ generatedDog: `data:image/png;base64,${res.data.image}` })
  }

  render() {
    const { generatedDog } = this.state;
    return (
      <div>
        <h1>Doggin' Dog GAN</h1>
        <button onClick={this.generateDog}>Generate A Dog</button>
        <button>Gannify This Dog</button>
        <img src={generatedDog} alt="generated dog"/>
      </div>
    );
  }
}

export default App;
