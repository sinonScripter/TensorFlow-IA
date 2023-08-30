  import { StatusBar } from 'expo-status-bar';
  import { Image, View , ActivityIndicator } from 'react-native';
  import { styles } from './styles';
  import React, { useState } from 'react';
  import { Button } from './components/Button';
  import * as ImagePicker from 'expo-image-picker';
  import * as tensorflow from '@tensorflow/tfjs';
  import * as mobilenet from '@tensorflow-models/mobilenet';
  import * as FileSystem from 'expo-file-system';
  import { decodeJpeg } from '@tensorflow/tfjs-react-native';
  import { Classification, ClassificationProps } from './components/Classification';


  export default function App() {
    const [isLoading, setIsLoading] = useState(false);
    const [selectedImageUri, setSelectedImageUri] = useState('');
    const [results, setResult] = useState<ClassificationProps[]>([]);

    async function handleSelectImage(){
      setIsLoading(true);

      try{
        const results = await ImagePicker.launchImageLibraryAsync({
          mediaTypes: ImagePicker.MediaTypeOptions.Images,
          allowsEditing: true
        }); 

        if (!results.canceled){
          const { uri } = results.assets[0];
          setSelectedImageUri( uri );
          await imageClassification(uri);
        }

      } catch (error){
        console.log(error);
      }finally{
      setIsLoading(false);
      }

    }
    
  async function imageClassification(imageUri: string) {
    setResult([]);
    
    await tensorflow.ready();
    const model = await mobilenet.load();

    const imageBase64 = await FileSystem.readAsStringAsync(imageUri, {
      encoding:FileSystem.EncodingType.Base64
    });

    const imgBuffer = tensorflow.util.encodeString(imageBase64 , 'base64').buffer;
    const raw = new Uint8Array(imgBuffer);
    const imageTensor = decodeJpeg(raw);

    const clasificationResult = await model.classify(imageTensor);
    setResult (clasificationResult)

  }

    return (
      <View style={styles.container}>
        
        <StatusBar 
        style="light"
        backgroundColor='transparent' 
        translucent
        />

        <Image
        source={ {uri:selectedImageUri ? selectedImageUri: 'https://imgs.search.brave.com/udmDGOGRJTYO6lmJ0ADA03YoW4CdO6jPKGzXWvx1XRI/rs:fit:860:0:0/g:ce/aHR0cHM6Ly90My5m/dGNkbi5uZXQvanBn/LzAyLzY4LzU1LzYw/LzM2MF9GXzI2ODU1/NjAxMl9jMVdCYUtG/TjVyalJ4UjJleVYz/M3puSzRxblllS1pq/bS5qcGc' }}
        style = {styles.image}
        />

        <View style={styles.results}>

           {
            results.map((result) => (
            <Classification key={result.className} data={result}/>
            ))
          }
  
        </View>

        {
          isLoading
          ? <ActivityIndicator color="royalblue"/> 
          : <Button title="Selecionar Imagem" onPress={handleSelectImage}/>}
    
      </View>
    );
  }

