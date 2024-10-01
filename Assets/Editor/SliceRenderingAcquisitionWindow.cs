using System.IO;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;


using Unity.Collections;
using Unity.Mathematics;
using UnityEngine.Rendering;

namespace UnityVolumeRendering
{
    public class SliceRenderingAcquisitionWindow : EditorWindow
    {
        private int selectedPlaneIndex = -1;


        [MenuItem("Volume Rendering/Slice acquisition")]
        
        static void ShowWindow()
        {
            SliceRenderingAcquisitionWindow wnd = new SliceRenderingAcquisitionWindow();
            wnd.Show();
            wnd.SetInitialPosition();
            wnd.Focus();
        }

        private void SetInitialPosition()
        {
            Rect rect = this.position;
            Debug.Log($" SetInitialPosition {this.position}");
            rect.width = 800.0f;
            rect.height = 500.0f;
            this.position = rect;
        }

        private void OnFocus()
        {
            // set selected plane as active GameObject in Hierarchy
            SlicingPlane[] spawnedPlanes = FindObjectsOfType<SlicingPlane>();

            if (selectedPlaneIndex != -1 && spawnedPlanes.Length > 0)
            {
                Selection.activeGameObject = spawnedPlanes[selectedPlaneIndex].gameObject;
            }
        }

        private void OnGUI()
        {
            SlicingPlane[] spawnedPlanes = FindObjectsOfType<SlicingPlane>();

            if (spawnedPlanes.Length > 0)
                selectedPlaneIndex = selectedPlaneIndex % spawnedPlanes.Length;

            float bgWidth = Mathf.Min(this.position.width, (this.position.height * 2.0f));
            // Rect bgRect = new Rect(0.0f, 0.0f, bgWidth, bgWidth);
            Rect bgRect = new Rect(0.0f, 0.0f, bgWidth * 0.5f, bgWidth * 0.5f);
            // Rect newRect = new Rect(0.0f, 0.0f, bgWidth* 0.5f, bgWidth * 0.5f);
            // float bgWidth = Mathf.Min(this.position.width - 20.0f, (this.position.height - 50.0f) * 2.0f);
            // Rect bgRect = new Rect(0.0f, 0.0f, bgWidth, bgWidth * 0.5f);
            //Debug.Log("OnGUI");
            //Debug.Log($" bgrect {bgRect}");
            //Debug.Log($" s plane index {selectedPlaneIndex}");
            //Debug.Log(spawnedPlanes.Length);

            if (selectedPlaneIndex != -1 && spawnedPlanes.Length > 0)
            {
                SlicingPlane planeObj = spawnedPlanes[System.Math.Min(selectedPlaneIndex, spawnedPlanes.Length - 1)];
                // Draw the slice view
                Material mat = planeObj.GetComponent<MeshRenderer>().sharedMaterial;
                Graphics.DrawTexture(bgRect, mat.GetTexture("_DataTex"), mat);

//                GUIUtility.RotateAroundPivot(180.0f, new Vector2(bgRect.width * 0.5f, bgRect.height * 0.5f)); // removed plane rotation
                //Debug.Log($"type { mat.GetTexture("_DataTex").GetType().ToString()            }");
                //Debug.Log($" new image {System.Math.Min(selectedPlaneIndex, spawnedPlanes.Length - 1)}");
                
                //Debug.Log($" mat {mat}");

                // Graphics.DrawTexture(bgRect, mat.GetTexture("_DataTex"), mat);
                
                //Texture tex = mat.GetTexture("_DataTex"); // tex.dimension daje Tex3D, ale nie można użyć tex.depth bo mówi że to Texture a nie Texture3D
                // Debug.Log($" tex {tex}");
                //Debug.Log($" tex {tex.height}");
                //Debug.Log($" tex {tex.width}");
                // // Debug.Log($" tex {tex.depth}");

                // Debug.Log($" tex {tex.dimension}");
                // Debug.Log($" tex {tex.graphicsFormat}");


                Texture3D tex2 = (Texture3D)mat.GetTexture("_DataTex");

                // Debug.Log($" tex2 {tex2}");
                // Debug.Log($" tex2 {tex2.dimension}");
                // Debug.Log($"type2 { tex2.GetType().ToString()            }");
                // Debug.Log($" tex2 {tex2.height}");
                // Debug.Log($" tex2 {tex2.width}");
                // Debug.Log($" tex2 {tex2.depth}");
                // Debug.Log($" tex2 {tex2.graphicsFormat}");

                RenderTexture rtDestination = new(tex2.width, tex2.height, 0, RenderTextureFormat.ARGB32);
                rtDestination.Create();
                Graphics.Blit(mat.GetTexture("_DataTex"), rtDestination, mat);


                // RenderTexute to Texture2D


                Texture2D tex2d = new Texture2D(tex2.width, tex2.height, TextureFormat.RGB24, false);
                var old_rt = RenderTexture.active;
                RenderTexture.active = rtDestination;

                tex2d.ReadPixels(new Rect(0, 0, rtDestination.width, rtDestination.height), 0, 0);
                tex2d.Apply();

                RenderTexture.active = old_rt;

                // Debug.Log($" tex2d {tex2d}");


                // to array


                NativeArray<float4> data = new NativeArray<float4>(tex2d.height * tex2d.width, Allocator.Persistent);
                var request = AsyncGPUReadback.RequestIntoNativeArray(ref data, tex2d);
                request.WaitForCompletion();


                float[,] transformed = new float[tex2d.height,  tex2d.width];
                int counter = 0;
                for (int y = 0; y < tex2d.width; y++)
                {
                    for (int x = 0; x < tex2d.height ; x++)
                    {
                        float4 val = data[counter];
                        counter++;

                        transformed[x,y] = val.x;
                    }
                }
                data.Dispose();
                // Debug.Log($" transformed {transformed}");
                // Debug.Log($" transformed {transformed[4,4]}");
                // File.WriteAllBytes("transformed.txt", transformed.ToArray());

				var bytes = tex2d.EncodeToPNG();
				DestroyImmediate(tex2d);
				// File.WriteAllBytes("AcquiredData/pic.png", bytes);















                // request.WaitForCompletion();

                // float[,] transformed = new float[us.mainRaysCount, us.maxRows];
                // int counter = 0;
                // for (int y = 0; y < us.maxRows; y++)
                // {
                //     for (int x = 0; x < us.mainRaysCount; x++)
                //     {
                //         float4 val = data[counter];
                //         counter++;

                //         transformed[x,y ] = val.x;
                //     }
                // }
                // data.Dispose();







                // ============================

                // Texture2D tex2 = new Texture2D(tex.width, tex.height, TextureFormat.RGB24, false);
                // Texture3D = 



                // ================================

                //Texture2D tex2D = (tex as Texture2D); https://discussions.unity.com/t/convert-texture-to-texture2d/203896/2
                // Texture2D tex2D = tex.ToTexture2D();
                //Texture2D tex2D = (Texture2D)tex;

                //byte[] bytes = ImageConversion.EncodeToJPG(tex2D);

                // var data = tex2D.GetRawTextureData<Color32>();
                // Color[] pixels = tex.GetPixels(0);


                // Debug.Log($" pixels {pixels}");

                // ========================== https://stackoverflow.com/questions/44264468/convert-rendertexture-to-texture2d


                // Texture2D tex2 = new Texture2D(tex.width, tex.height, TextureFormat.RGB24, false);
                // var old_rt = RenderTexture.active;
                // RenderTexture.active = tex;

                // tex2.ReadPixels(new Rect(0, 0, tex.width, tex.height), 0, 0);
                // tex2.Apply();

                // RenderTexture.active = old_rt;

                // Debug.Log($" tex2 {tex2}");

                // ==============================



                // NativeArray<float4> data = new NativeArray<float4>(tex.height * tex.width, Allocator.Persistent);
                //var request = AsyncGPUReadback.RequestIntoNativeArray(ref data, tex);
                // request.WaitForCompletion();


                // float[,] transformed = new float[tex.height,  tex.width];
                // int counter = 0;
                // for (int y = 0; y < tex.width; y++)
                // {
                //     for (int x = 0; x < tex.height ; x++)
                //     {
                //         float4 val = data[counter];
                //         counter++;

                //         transformed[x,y ] = val.x;
                //     }
                // }
                // data.Dispose();
                // Debug.Log($" transformed {transformed}");

                // request.WaitForCompletion();

                // float[,] transformed = new float[us.mainRaysCount, us.maxRows];
                // int counter = 0;
                // for (int y = 0; y < us.maxRows; y++)
                // {
                //     for (int x = 0; x < us.mainRaysCount; x++)
                //     {
                //         float4 val = data[counter];
                //         counter++;

                //         transformed[x,y ] = val.x;
                //     }
                // }
                // data.Dispose();




            }

        }




        public void OnInspectorUpdate()
        {
            Repaint();
        }
    }
}