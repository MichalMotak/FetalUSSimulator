using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.IO;

using AustinHarris.JsonRpc;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Rendering;

namespace UnityVolumeRendering
{
    
    public class Rpc : JsonRpcService {

        SlicingPlane plane;
        SavePlaneMovementRot saveplanemovepos;


        public Rpc(SlicingPlane plane, SavePlaneMovementRot saveplanemovepos){
            this.plane = plane;
            this.saveplanemovepos = saveplanemovepos;
        }

        public void speak(){
            Debug.Log("RPC Speaks");
        }


        [JsonRpcMethod]
        public void Say(string message){
            Debug.Log($"you sent {message}");
        }
        
        [JsonRpcMethod]
        float[] GetPosition(){
            

            float value = 16;
            Debug.Log($" sent {value}");

            //Debug.Log($" plane 2{plane.transform.position.y}");

            float[] pos = new float[3];
            pos[0] = plane.transform.position.x;
            pos[1] = plane.transform.position.y;
            pos[2] = plane.transform.position.z;

            Debug.Log($" sending vector {plane.transform.position}");
            Debug.Log($" sending vector {pos}");

            return pos;
            //return plane.transform.position.y;
        }


        
        [JsonRpcMethod]
        float[,] GetImage(){
            

            // Debug.Log($" sent {value}");

            float[,] pos = new float[2,2];
            pos[0,0] = 1;
            pos[0,1] = 1;
            pos[1,0] = 3;
            pos[1,1] = 4;

            Debug.Log($" sending vector 2 {pos}");

            return pos;
            //return plane.transform.position.y;
        }


        [JsonRpcMethod]
        public (int[], byte[], int, string[]) GetOutput(){  // float[,] for raw data
            
            // int[] shape = new int[2];
            // shape[0] = 2;
            // shape[1] = 2;

            // Debug.Log($" sent {value}");

            // float[,] pos = new float[2,2];
            // pos[0,0] = 1;
            // pos[0,1] = 1;
            // pos[1,0] = 3;
            // pos[1,1] = 4;

            // Debug.Log($" sending vector 2 {pos}");



            Material mat = this.plane.GetComponent<MeshRenderer>().sharedMaterial;


            Texture3D tex2 = (Texture3D)mat.GetTexture("_DataTex");

            // Debug.Log($" tex2 {tex2}");
            // Debug.Log($" tex2 {tex2.dimension}");
            // Debug.Log($"type2 { tex2.GetType().ToString()            }");
            // Debug.Log($" tex2 {tex2.height}");
            // Debug.Log($" tex2 {tex2.width}");
            // Debug.Log($" tex2 {tex2.depth}");
            // Debug.Log($" tex2 {tex2.graphicsFormat}");

            RenderTexture rtDestination = new(tex2.width, tex2.height, 0, RenderTextureFormat.ARGB32); //ARGB32, R8
            rtDestination.Create();
            Graphics.Blit(mat.GetTexture("_DataTex"), rtDestination, mat);


            // RenderTexute to Texture2D


            Texture2D tex2d = new Texture2D(tex2.width, tex2.height, TextureFormat.RGB24, false); //GB24, Alpha8
            var old_rt = RenderTexture.active;
            RenderTexture.active = rtDestination;

            tex2d.ReadPixels(new Rect(0, 0, rtDestination.width, rtDestination.height), 0, 0);
            tex2d.Apply();

            RenderTexture.active = old_rt;

            // Debug.Log($" tex2d {tex2d}");


            // to array raw values 

            // NativeArray<float4> data = new NativeArray<float4>(tex2d.height * tex2d.width, Allocator.Persistent);
            // var request = AsyncGPUReadback.RequestIntoNativeArray(ref data, tex2d);
            // request.WaitForCompletion();


            // float[,] transformed = new float[tex2d.height,  tex2d.width];
            // int counter = 0;
            // for (int y = 0; y < tex2d.width; y++)
            // {
            //     for (int x = 0; x < tex2d.height ; x++)
            //     {
            //         float4 val = data[counter];
            //         counter++;

            //         transformed[x,y] = val.x;
            //     }
            // }
            // data.Dispose();
            // Debug.Log($" transformed {transformed}");


            // Convert to JPEG 
            int[] shape = {tex2.width, tex2.height};

            byte[] bytes = ImageConversion.EncodeToPNG(tex2d);


            // File.WriteAllBytes("AcquiredData/imagejpg.jpg", bytes);
            // File.WriteAllBytes("AcquiredData/bytesGray8.txt", bytes);

            // File.WriteAllBytes("AcquiredData/imageGraypng.png", bytes);
            // File.WriteAllBytes("AcquiredData/bytesGray8_png.txt", bytes);




            Object.Destroy(tex2d);
            // id++;

            // for (int y = 0; y < 10; y++)
            // {
            //     Debug.Log($" bytes {bytes[y]}");
            // }

            // float[,] transformed = new float[tex2d.height,  tex2d.width];
            // int counter = 0;
            // for (int y = 0; y < tex2d.width; y++)
            // {
            //     for (int x = 0; x < tex2d.height ; x++)
            //     {
            //         byte val = bytes[counter];
            //         counter++;

            //         transformed[x,y] = val;
            //     }
            // }

            // Debug.Log($" transformed {transformed}");
            // for (int y = 0; y < 10; y++)
            //     {
            //         Debug.Log($" transformed {transformed[y,0]}");
            //     }


            return (shape, bytes, saveplanemovepos.screenshotIndex, saveplanemovepos.rowDataTemp);
            //return plane.transform.position.y;
        }
    }








    public class p_communication : MonoBehaviour
    {
        private SlicingPlane slicingPlane;
        private SavePlaneMovementRot saveplanemovepos;

        public int x = 4;


        public static p_communication Instance { get; private set; }

        public static Rpc rpc { get; private set; }

        // void Awake()
        // {
        //     // Ensure that there's only one instance of ClassA
        //     if (Instance == null)
        //     {
        //         Instance = this;
        //         DontDestroyOnLoad(gameObject); // Optional: Keep the instance between scenes
        //     }
        //     else
        //     {
        //         Destroy(gameObject);
        //     }
        // }


        // Start is called before the first frame update
        void Start()
        {

            slicingPlane = GameObject.FindObjectOfType<SlicingPlane>();
            saveplanemovepos = FindObjectOfType<SavePlaneMovementRot>();

            rpc = new Rpc(slicingPlane, saveplanemovepos);

			// Debug.Log($" ind start {saveplanemovepos.screenshotIndex}");
            
        }

        // Update is called once per frame
        void Update()
        {
            
        }


        public void speak(){
            Debug.Log("RPC Speaks");
        }



        void OnEnable()
        {
            SavePlaneMovementRot.OnFunctionB += ExecuteFunctionB;
        }

        void OnDisable()
        {
            SavePlaneMovementRot.OnFunctionB -= ExecuteFunctionB;
        }

        void ExecuteFunctionB()
        {
            Debug.Log("Function B is running after Function A");
            rpc.speak();
            rpc.Say("ac");
            rpc.GetOutput();
        }




    }



}