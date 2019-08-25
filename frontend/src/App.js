import React from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import Footer from './Footer.jsx';

class App extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      generatedDog: null,
      realDog: null,
      ganDog: null,
      gannifyClicked: false,
    }
  }

  retrieveDog = async () => {
    const res = await axios.get('/api/images') 
    this.setState({ 
      gannifyClicked: false, 
      realDog: `data:image/jpg;base64,${res.data.image}` 
    })
  }
  
  generateDog = async () => {
    const res = await axios.get('/api/generate') 
    this.setState({ generatedDog: `data:image/png;base64,${res.data.image}` })
  }

  gannifyDog = async () => {
    const res = await axios.post('/api/gannify', {
      dog: this.state.realDog.substr(22), 
    }) 
    this.setState({ 
      gannifyClicked: true, 
      ganDog: `data:image/png;base64,${res.data.image}` 
    })
  }

  showRealOrGanDog = () => {
    const { gannifyClicked, ganDog, realDog } = this.state;

    let dogToDisplay = <img className="dog-img" src={realDog} alt="real dog" />
    if (gannifyClicked) {
      dogToDisplay = <img className="dog-img" src={ganDog} alt="gannified dog" />
    }

    return (
      <div>
        {dogToDisplay}
      </div>
    )
  }

  renderGanButton = () => {
    const { gannifyClicked } = this.state;

    if (gannifyClicked) {
      return <button onClick={this.retrieveDog}>Gannify Another Dog?</button>  
    } else {
      return <button onClick={this.gannifyDog}>Gannify This Dog</button>  
    }
  }

  showGeneratedDog = () => {
    const { generatedDog } = this.state;

    if (generatedDog) {
      return (
        <div>
          <img className="dog-img" src={generatedDog} alt="generated dog"/>
        </div>
      )
    }
    return (
      <div>
        <img
          src="https://res.cloudinary.com/teepublic/image/private/s--SATpAoiT--/t_Preview/b_rgb:191919,c_limit,f_jpg,h_630,q_90,w_630/v1466467225/production/designs/554892_1.jpg"
          alt="dog emoji"
          className="dog-img" />
      </div>
    )
  }

  componentDidMount() {
    this.retrieveDog();
  }

  render() {
    return (
      <div>
        <div style={{ marginBottom: 25 }} className="f-jcc f-aic">
          <h1>Doggin' Dog GAN</h1>
        </div>
        <div className="flex-col">
          <div className="flex-col" style={{ marginBottom: 25 }}>
            {this.showGeneratedDog()}
            <button onClick={this.generateDog}>Generate A Dog</button>
          </div>
          <div className="flex-col">
            <div>
              {this.showRealOrGanDog()}
            </div>
            {this.renderGanButton()}
          </div>
          <h2 style={{ marginTop: 35, marginBottom: 5 }}>Curious how it works?</h2>
          <Link to="/about">
            <button>About</button>
          </Link>
          <Footer />
        </div>
      </div>
    );
  }
}

export default App;
