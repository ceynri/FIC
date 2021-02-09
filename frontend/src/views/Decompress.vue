<template>
  <section class="decompressor">
    <PageFrame>
      <header class="title_wrapper">
        <h1 class="title">Decompressor</h1>
        <div class="comment">Upload compressed images to decompress</div>
      </header>
      <Uploader accept=".fic" />
    </PageFrame>
  </section>
</template>

<script>
import PageFrame from '@/components/PageFrame.vue';
import Uploader from '@/components/Uploader.vue';

export default {
  components: {
    PageFrame,
    Uploader,
  },
};
</script>

<style lang="scss" scoped>
.decompressor {
  .title {
    font-size: 32px;
    margin: 0;
    margin-bottom: 8px;
  }

  .comment {
    color: var(--text2);
    font-size: 20px;
    margin-bottom: 16px;
  }
}
</style>
